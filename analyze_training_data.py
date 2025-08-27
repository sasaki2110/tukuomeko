import json

def analyze_data(file_path, max_length=512):
    """
    指定されたJSONLファイル内のテキストデータの文字長を分析します。

    Args:
        file_path (str): 分析するJSONLファイルのパス。
        max_length (int): 最大文字長。これを超えるデータの件数をカウントします。
                          デフォルトは512です。

    Returns:
        dict: 総データ件数と最大文字長を超えるデータ件数を含む辞書。
              ファイルが見つからない場合はNoneを返します。
    """
    total_count = 0
    over_limit_count = 0
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                total_count += 1
                try:
                    data = json.loads(line)
                    # データの構造に合わせて'text'キーを調整してください
                    text_content = data.get('text', '')
                    if len(text_content) > max_length:
                        over_limit_count += 1
                except json.JSONDecodeError:
                    print(f"警告: 不正なJSON形式の行が見つかりました: {line.strip()}")
                except KeyError:
                    print(f"警告: 'text'キーが見つかりません: {line.strip()}")
    except FileNotFoundError:
        print(f"エラー: ファイル '{file_path}' が見つかりません。")
        return None

    return {
        "総データ件数": total_count,
        f"文字長が{max_length}を超える件数": over_limit_count
    }

# 使用例
if __name__ == "__main__":
    # ここに学習データのファイルパスを指定してください
    training_data_path = './new_recipe_data.jsonl'
    
    analysis_result = analyze_data(training_data_path, max_length=512)

    if analysis_result:
        print("--- 分析結果 ---")
        print(f"総データ件数: {analysis_result['総データ件数']} 件")
        print(f"文字長が512を超える件数: {analysis_result['文字長が512を超える件数']} 件")
        print("--------------")