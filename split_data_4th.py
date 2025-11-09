import random
import json

# ファイルパス
input_file = "4th_recipe_data.jsonl"
train_file = "4th_recipe_data_train.jsonl"
validation_file = "4th_recipe_data_validation.jsonl"

# 分割割合 (80:20)
validation_ratio = 0.2

# データを読み込む
with open(input_file, "r", encoding="utf-8") as f:
    data = f.readlines()

# データをシャッフル
random.shuffle(data)

# 分割点を計算
validation_split = int(len(data) * validation_ratio)
train_data = data[validation_split:]
validation_data = data[:validation_split]

# トレーニングデータを書き出す
with open(train_file, "w", encoding="utf-8") as f:
    f.writelines(train_data)

# 検証データを書き出す
with open(validation_file, "w", encoding="utf-8") as f:
    f.writelines(validation_data)

print(f"データ分割完了: 学習データ {len(train_data)}件, 検証データ {len(validation_data)}件")

