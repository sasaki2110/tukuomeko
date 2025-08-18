import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import sys

# FastAPIアプリケーションの初期化
app = FastAPI()

# CORS設定：WEBアプリコンテナからのアクセスを許可
# 本番環境では、より具体的なオリジンを指定することを推奨します。
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # すべてのオリジンを許可 (開発用)
    allow_credentials=True,
    allow_methods=["*"],  # すべてのHTTPメソッドを許可
    allow_headers=["*"],  # すべてのHTTPヘッダーを許可
)

# グローバル変数としてモデルとトークナイザーを保持
# アプリケーション起動時に一度だけロードし、リクエストごとに再ロードしないようにします
tokenizer = None
model = None

# モデルのロードを行う関数
# FastAPIのライフサイクルイベントとしてアプリケーション起動時に実行されます
@app.on_event("startup")
async def load_model():
    global tokenizer, model
    try:
        model_id = "google/gemma-2b-it"
        lora_adapter_path = "./gemma-finetuned" # LoRAアダプターのパス

        print(f"モデル '{model_id}' とトークナイザーをロード中...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        # パディングトークンがないモデルの場合に設定
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # bfloat16でのモデルロード
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto", # GPUが利用可能な場合は自動的にGPUにロード
            torch_dtype=torch.bfloat16,
        )

        # ファインチューニングしたLoRAアダプターをロード
        print(f"LoRAアダプター '{lora_adapter_path}' をロード中...")
        model = PeftModel.from_pretrained(model, lora_adapter_path)
        model.eval() # モデルを評価モードに設定
        print("モデルとLoRAアダプターのロードが完了しました。")

    except Exception as e:
        print(f"モデルのロード中にエラーが発生しました: {e}", file=sys.stderr)
        # エラーが発生した場合は、アプリケーションが起動しないようにするか、
        # サービスが利用できないことを示す状態を設定することを検討してください。

# 推論を実行するAPIエンドポイント
@app.post("/generate")
async def generate_response_api(request: Request):
    if tokenizer is None or model is None:
        return JSONResponse(status_code=503, content={"error": "モデルがまだロードされていません。"})

    try:
        data = await request.json()
        prompt_text = data.get("prompt")

        if not prompt_text:
            return JSONResponse(status_code=400, content={"error": "プロンプトが提供されていません。"})

        # プロンプトの形式を学習時と一致させる
        #full_prompt = prompt_text + "[回答]\n"
        full_prompt = prompt_text
        
        # プロンプトをトークン化
        inputs = tokenizer(
            full_prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        
        # GPUへの移動は、`device_map`によってモデルが自動的に行いますが、念のため明示的に指定
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # モデルで生成
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=256,
                do_sample=True,
                top_p=0.9,
                temperature=0.7,
                pad_token_id=tokenizer.pad_token_id, # 推論時のパディングトークンを指定
                eos_token_id=tokenizer.eos_token_id,
            )

        # 入力プロンプト部分を除外してデコード
        response_tokens = outputs[0][len(inputs["input_ids"][0]):]
        full_response = tokenizer.decode(response_tokens, skip_special_tokens=True)
        
        # 応答の最初の行のみを返却
        first_line_response = full_response.split('\n')[0].strip()

        return JSONResponse(content={"response": first_line_response})

    except Exception as e:
        print(f"推論中にエラーが発生しました: {e}", file=sys.stderr)
        return JSONResponse(status_code=500, content={"error": f"推論中にエラーが発生しました: {str(e)}"})

# アプリケーションを起動するためのコマンド (uvicorn)
# このスクリプトを直接実行する際にはこのブロックは実行されません
# コンテナ内で `uvicorn app:app --host 0.0.0.0 --port 8000` で実行します
if __name__ == "__main__":
    # この部分は、開発時に直接スクリプトを実行する場合にのみ使用されます。
    # 通常はuvicornコマンドで起動します。
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

