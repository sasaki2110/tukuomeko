import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer,
    BitsAndBytesConfig
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
import os

# --- 1. モデルとトークナイザーのロード ---
base_model_id = "google/gemma-2b-it"
pretrained_lora_path = "./gemma-finetuned"  # 前回学習したLoRAアダプタのパス

tokenizer = AutoTokenizer.from_pretrained(base_model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16
)

# --- 2. LoRAアダプターのロードとモデルの準備 ---
model = prepare_model_for_kbit_training(model)

# LoRA アダプタのみ読み込み（optimizer/scheduler state は復元しない）
model = PeftModel.from_pretrained(
    model,
    pretrained_lora_path,
    is_trainable=True  # ←これが重要：再学習可能にする
)

# 勾配チェックポイント有効化
model.gradient_checkpointing_enable()
model.print_trainable_parameters()

# --- 3. 新しいデータセットのロード ---
new_train_data_path = "./new_recipe_data_train.jsonl"
new_validation_data_path = "./new_recipe_data_validation.jsonl"

dataset = load_dataset(
    "json",
    data_files={
        "train": new_train_data_path,
        "validation": new_validation_data_path
    }
)

# --- 4. データの前処理 ---
def tokenize_function(examples):
    max_length = 512

    try:
        prompts_and_answers = [text.split('[回答]\n', 1) for text in examples["text"]]
        prompts = [item[0] for item in prompts_and_answers]
        answers = [item[1] for item in prompts_and_answers]
    except IndexError:
        raise ValueError("A sample in the dataset does not contain the '[回答]\\n' separator.")

    prompt_token_lengths = [len(tokenizer.encode(p, add_special_tokens=False)) for p in prompts]

    full_texts = [p + '[回答]\n' + a for p, a in zip(prompts, answers)]
    tokenized_output = tokenizer(
        full_texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )

    labels = [l.copy() for l in tokenized_output["input_ids"]]

    for i in range(len(labels)):
        mask_len = prompt_token_lengths[i] + 1
        separator_tokens = tokenizer.encode('[回答]\n', add_special_tokens=False)
        mask_len += len(separator_tokens)

        for j in range(mask_len):
            if j < len(labels[i]):
                labels[i][j] = -100

    tokenized_output["labels"] = labels
    return tokenized_output

tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"],
    num_proc=os.cpu_count(),
    desc="Tokenizing and masking data"
)

# --- 5. 学習設定 ---
training_args = TrainingArguments(
    output_dir="./gemma-finetuned-continued",
    overwrite_output_dir=True,
    num_train_epochs=2,  # ★ まずは短く
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=5e-5,  # ★ LoRA向けに調整
    warmup_ratio=0.03,   # ★ 追加
    lr_scheduler_type="cosine",  # ★ 追加
    fp16=False,
    bf16=True,
    logging_dir='./logs_continued',
    logging_steps=10,
    logging_first_step=True,  # ★ 最初のステップもログ
    eval_strategy="epoch",   # ← ここを修正
    save_strategy="epoch",
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    report_to="none"
)

# --- 6. Trainerの初期化と学習の開始 ---
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
)

# use_cache の警告を避ける
model.config.use_cache = False

# --- 7. 学習の実行 ---
print("Train dataset size:", len(tokenized_dataset["train"]))
print("Eval dataset size:", len(tokenized_dataset["validation"]))

steps_per_epoch = len(tokenized_dataset["train"]) // (
    training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
)
print("Steps per epoch:", steps_per_epoch)
print("Total optimization steps:", steps_per_epoch * training_args.num_train_epochs)

train_result = trainer.train()

# --- 8. モデルの保存 ---
trainer.save_model("./gemma-finetuned-continued")

# --- 9. 学習結果を表示 ---
metrics = train_result.metrics
print("Training completed. Metrics:", metrics)
