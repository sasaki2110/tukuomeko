import torch, json
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

base_model_id = "google/gemma-2b-it"
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)

tokenizer = AutoTokenizer.from_pretrained(base_model_id)
model_base = AutoModelForCausalLM.from_pretrained(base_model_id, quantization_config=bnb_config, device_map="auto", torch_dtype=torch.bfloat16)

# 追加学習結果のディレクトリ
#finetuned_dir = "./gemma-finetuned-continued"
#finetuned_dir = "./gemma-finetuned"
finetuned_dir = "./gemma-finetuned-new-continued"
#finetuned_dir = "./gemma-finetuned-new"
#finetuned_dir = "./gemma-finetuned-base"
model = PeftModel.from_pretrained(model_base, finetuned_dir)
model.eval()
model.config.use_cache = True

def infer(prompt, max_new_tokens=256):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, num_beams=1, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0], skip_special_tokens=True)

# 例: dataset の先頭 10 件を検証用にロードする（ローカル jsonl 等に合わせて）
samples = []
with open("./new_recipe_data_validation.jsonl","r") as f:
    for i, line in enumerate(f):
        if i>=10: break
        samples.append(json.loads(line)["text"])

for s in samples:
    prompt = s.split('[回答]\n',1)[0] + "[回答]\n"
    out = infer(prompt)
    print("=== PROMPT ===")
    print(prompt)
    print("=== OUTPUT ===")
    print(out)
    print("\n")
