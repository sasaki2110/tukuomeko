# 環境
apt-get update
apt-get install -y python3 python3-pip
pip install torch transformers accelerate peft bitsandbytes datasets
pip install fastapi uvicorn

# 実行
uvicorn app:app --host 0.0.0.0 --port 8000 &

