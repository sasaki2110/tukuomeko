import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel # PeftModelをインポート
from transformers import BitsAndBytesConfig
import os

# Hugging Faceのキャッシュディレクトリを設定 (任意: デフォルトはユーザーのホームディレクトリ)
# os.environ['HF_HOME'] = '/app/hf_cache'

# --- 1. モデルとトークナイザーのロード ---
# ベースモデルのID (Gemma 2B Instruction-tuned)
base_model_id = "google/gemma-2b-it"
# 前回学習したLoRAアダプターが保存されているディレクトリのパス
# 前回の学習スクリプトで `trainer.save_model("./gemma-finetuned")` と保存した場合、このパスになります。
pretrained_lora_path = "./gemma-finetuned"

# トークナイザーのロード (ベースモデルからロード)
tokenizer = AutoTokenizer.from_pretrained(base_model_id)
# パディングトークンがないモデルの場合に設定
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token # EOSトークンをパディングトークンとして設定

# 4-bit 量子化設定 (前回と同じ設定を再適用)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4", # NormalFloat 4-bit 量子化タイプ
    bnb_4bit_use_double_quant=True, # 二重量子化を使用
    bnb_4bit_compute_dtype=torch.bfloat16, # 計算時のデータ型
)

# ベースモデルを量子化設定でロード
# これにより、Gemma 2B ITの基本構造がロードされ、量子化が適用されます。
model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16 # モデルのロード時のデータ型 (量子化と合わせてbf16推奨)
)

# --- 2. LoRAアダプターのロードとモデルの準備 ---
# 量子化されたベースモデルをLoRAトレーニング用に準備します。
# これにより、PEFTモジュールが有効になり、アダプターが結合できるようになります。
model = prepare_model_for_kbit_training(model)

# 以前にファインチューニングしたLoRAアダプターをロードします。
# これにより、ロードしたベースモデルの上に以前のLoRA重みが適用されます。
# これが追加学習の肝となります。
model = PeftModel.from_pretrained(model, pretrained_lora_path)

# LoRA設定 (既存のアダプターを継続して学習するため、この設定は主に情報として残しますが、
# 既存のアダプターのR, alphaなどはロード時に自動で引き継がれます。)
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj", "o_proj", "k_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
# 注意: PeftModel.from_pretrainedで既にアダプターをロードしているため、
# get_peft_modelを再度呼び出す必要はありません。モデルは既にPeftModelインスタンスです。

# LoRAが適用されたモデルの訓練可能なパラメータ数を表示
# 追加学習でも、LoRAアダプターのパラメータのみが学習対象となります。
model.print_trainable_parameters()


# --- 3. 新しいデータセットのロード ---
# ここに**新しい学習データと検証データ**のパスを指定してください。
# 例として、ファイル名を変更しています。必要に応じて実際のパスに変更してください。
new_train_data_path = "/app/data/new_recipe_data_train.jsonl"
new_validation_data_path = "/app/data/new_recipe_data_validation.jsonl"

dataset = load_dataset(
    "json",
    data_files={
        "train": new_train_data_path,
        "validation": new_validation_data_path
    }
)

# --- 4. データの前処理 (以前のスクリプトから変更なし) ---
def tokenize_function(examples):
    # トークン化の最大長を設定
    max_length = 512 

    # プロンプトと回答を分離
    # " [回答]\n" が存在しないデータはエラーになるので、データ形式を統一しておくことが重要
    try:
        # 'text'フィールドをプロンプトと回答に分割
        prompts_and_answers = [text.split('[回答]\n', 1) for text in examples["text"]]
        prompts = [item[0] for item in prompts_and_answers]
        answers = [item[1] for item in prompts_and_answers]
    except IndexError:
        raise ValueError("A sample in the dataset does not contain the '[回答]\\n' separator.")

    # プロンプト部分をトークン化し、その長さを取得
    # add_special_tokens=Falseで、<bos>などの特殊トークンの長さを除外
    prompt_token_lengths = [len(tokenizer.encode(p, add_special_tokens=False)) for p in prompts]
    
    # 全体のテキスト（プロンプト＋回答）をトークン化
    full_texts = [p + '[回答]\n' + a for p, a in zip(prompts, answers)]
    tokenized_output = tokenizer(
        full_texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )

    labels = [l.copy() for l in tokenized_output["input_ids"]]
    
    # 各データサンプルに対してループ
    for i in range(len(labels)):
        prompt_len = prompt_token_lengths[i]
        
        # モデルの入力には <bos> トークンが先頭に付与されるため、
        # マスクする長さはプロンプトのトークン長 + 1 となる
        mask_len = prompt_len + 1 # +1 は <bos> トークンの分

        # プロンプト部分と区切り文字部分のラベルを-100でマスク
        # [回答]\n のトークン長も考慮
        separator_tokens = tokenizer.encode('[回答]\n', add_special_tokens=False)
        mask_len += len(separator_tokens)

        # ラベルの先頭から mask_len 個を -100 に設定
        for j in range(mask_len):
            if j < len(labels[i]):
                labels[i][j] = -100
                
    tokenized_output["labels"] = labels
    return tokenized_output

# データセットにトークナイズ関数を適用
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True, # バッチ処理で効率化
    remove_columns=["text"], # 元の'text'列はもう不要
    num_proc=os.cpu_count(), # CPUコア数に応じて並列処理 (環境により調整)
    desc="Tokenizing and masking data" # 処理の進捗バーに表示される説明
)

# --- 5. 学習設定 (TrainingArguments) ---
# 追加学習のため、設定を調整します。
training_args = TrainingArguments(
    output_dir="./gemma-finetuned-continued", # 新しい追加学習モデルの保存先ディレクトリ
    overwrite_output_dir=True, # 既に存在する場合は上書き
    num_train_epochs=5, # 学習エポック数 (新しいデータの量に合わせて調整してください)
    per_device_train_batch_size=1, # GPUあたりのバッチサイズ (8GB VRAMでは1が安全)
    gradient_accumulation_steps=4, # 勾配蓄積ステップ数 (実質バッチサイズ: 1 * 4 = 4)
    learning_rate=1e-5, # **重要**: 追加学習では、通常、初期学習よりも小さい学習率に設定します。
    fp16=False, # bf16を既に設定しているので、ここではfp16=False
    bf16=True, # bfloat16を使用
    logging_dir='./logs_continued', # ログの保存先ディレクトリも新しいものにすると良いでしょう
    logging_steps=10, # ログ出力の頻度
    eval_strategy="epoch", # 各エポック終了ごとに検証を実行
    save_strategy="epoch", # 評価戦略と保存戦略を一致させる
    save_total_limit=1, # 保存するチェックポイントの最大数 (最も良いモデルの1つのみ)
    load_best_model_at_end=True, # 学習終了時に最も良いモデルをロード
    metric_for_best_model="eval_loss", # 最も良いモデルを判断するメトリック
    report_to="none" # ログ報告先 (wandb, tensorboardなど、今回はなし)
)

# --- 6. Trainerの初期化と学習の開始 ---
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"], # 訓練データセット
    eval_dataset=tokenized_dataset["validation"], # 検証データセット
    tokenizer=tokenizer, # Tokenizerを渡す
)

# 'use_cache=True' is incompatible with gradient checkpointing. Setting 'use_cache=False'.
# の警告を回避するため、Trainer初期化後にuse_cacheを明示的にFalseに設定
model.config.use_cache = False

trainer.train()

# --- 7. モデルの保存 ---
# 学習済みモデルを保存
# 追加学習後のモデルを新しいディレクトリに保存することを推奨します。
trainer.save_model("./gemma-finetuned-continued")
