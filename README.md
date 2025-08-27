# 環境
apt-get update
apt-get install -y python3 python3-pip
pip install torch transformers accelerate peft bitsandbytes datasets
pip install fastapi uvicorn

# 実行
uvicorn app:app --host 0.0.0.0 --port 8000 &

# スクリプト説明

analyze_training_data.py    学習データが512byte以下かチェック
app.py                      実際に推論し、推論結果を返すFastAPI
base_train.py               base 系モデルでのLoRA初期学習
base_train3.py              base 系モデルでのLoRA追加学習
new_train.py                it 系モデルでのLoRa初期学習（eos あり）
new_train3.py               it 系モデルでのLoRa追加学習（eos あり）
split_data.py               学習データを訓練用と検証用に分割
train.py                    it 系モデルでのLoRa初期学習（eos 無し　注）
train3.py                   it 系モデルでのLoRa初期学習（eos 無し　注）

注：eos の有無は学習データの問題の為、実際のロジックは同じ
　　出力するLoRA学習結果の出力フォルダが違うだけ。
