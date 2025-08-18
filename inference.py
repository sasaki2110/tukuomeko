import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# 1. ベースモデルとトークナイザーをロード
model_id = "google/gemma-2b-it"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# パディングトークンがないモデルの場合に設定
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# bfloat16でのモデルロード
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

# 2. ファインチューニングしたLoRAアダプターをロード
lora_adapter_path = "./gemma-finetuned"
model = PeftModel.from_pretrained(model, lora_adapter_path)
model.eval() # モデルを評価モードに設定

# 3. 推論を実行する関数
def generate_response(prompt):
    # プロンプトの形式を学習時と一致させる
    full_prompt = prompt + "[回答]\n"
    
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
    response = tokenizer.decode(response_tokens, skip_special_tokens=True)
    
    return response

# 4. 学習したタスクのプロンプトで試す
prompt_text = "以下のレシピの分類を行ってください。\n\n  [レシピ本文]\n(title)めっちゃジューシー♪鶏の唐揚げ (ingredientText)鶏もも肉300g ●おろしにんにく1片 ●おろししょうが1片 ●酒大さじ2 ●醤油大さじ1.1/2 ●ごま油小さじ1 卵1/2個 揚げ油適量 ◎薄力粉大さじ1.1/2 ◎片栗粉大さじ1.1/2\n"
#prompt_text = "以下のレシピの分類を行ってください。\n\n  [レシピ本文]\n  (title)簡単チキンナゲット♪～鶏胸肉で～ (ingredientText)鶏むね肉６００ｇ 卵１個 酒・醤油・マヨネーズ各大さじ１ 小麦粉大さじ６ 塩小さじ１弱 こしょう適量\n"
#prompt_text = "以下のレシピの分類を行ってください。\n\n  [レシピ本文]\n  (title)おいし～い☆うちの回鍋肉（ホイコーロー） (ingredientText)キャベツ大きめの葉4～5枚 ピーマン3個 豚バラ切り落とし200～300ｇ （下味用 酒大さじ2＋しょうゆ小さじ1/2） 片栗粉1/3カップ しょうが、にんにく各1切 ☆オイスターソース大さじ1～2 ☆豆板醤小さじ1/4 ☆しょうゆ小さじ1/2 ☆味噌小さじ1 ☆みりん大さじ1 ☆酒大さじ1 サラダ油・ごま油各大さじ1\n"
#prompt_text = "以下のレシピの分類を行ってください。\n\n  [レシピ本文]\n  (title)中華大好き♡　チンジャオロース春巻き (ingredientText)豚肉３００ｇ ピーマン３個 ゆでたけのこ１本 ●しょうゆ大１ ●酒大１ ●塩少々 片栗粉大１ ☆しょうゆ大１ ☆オイスターソース大３ ☆砂糖小１ ☆塩小１ ☆こしょう少々\n"
#prompt_text = "以下のレシピの分類を行ってください。\n\n  [レシピ本文]\n  (title)✽もやしと春雨の簡単春巻き✽ (ingredientText)豚ひき肉or合挽肉２５０g もやし１袋２００g はるさめ細いタイプ(小分けパック40g×2)８０g 春巻きの皮１０枚～ 塩こしょう少々 あわせ調味料 〇水８０cc ○料理酒５０cc 〇とりがらスープの素小さじ２ 〇オイスターソース大さじ１ 〇こいくち醤油大さじ１ 〇砂糖小さじ２ ○おろし生姜小さじ1 〇片栗粉大さじ１ 〇ゴマ油小さじ１ ★サラダ油適量\n"
#prompt_text = "以下のレシピの分類を行ってください。\n\n  [レシピ本文]\n  (title)白だしで作る親子丼 (ingredientText)鶏もも肉１００グラム ご飯茶碗１杯 玉葱１／４個分 卵１個 タレ 醤油大さじ１ 白だし大さじ１ みりん大さじ２ 酒大さじ１\n"
#prompt_text = "以下のレシピの分類を行ってください。\n\n  [レシピ本文]\n  (title)きのことベーコンの醤油風味クリームパスタ (ingredientText)パスタ麺２束 しめじ１パック エリンギ３本入り１パック ベーコン４枚 にんにく１片 生クリーム（ホイップ可）１００ｃｃ 牛乳３００ｃｃ 薄力粉大さじ２杯 コンソメ１個 醤油小さじ３杯 オリーブオイル大さじ２杯 塩コショウ適量\n"
#prompt_text = "以下のレシピの分類を行ってください。\n\n  [レシピ本文]\n  (title)こんなに簡単♪エビチリ (ingredientText)えび４００ｇ （下味） ・酒大さじ１ ・生姜汁少々 ・おろしにんにく少々 ・塩少々 にんにくみじんぎり１かけ ねぎみじんぎり大さじ４ サラダ油大さじ２ 豆板醤小さじ１～２ 酒大さじ２ ごま油大さじ１／２ （合わせ調味料） ・ケチャップ大さじ４ ・砂糖大さじ１ ・醤油大さじ１ ・片栗粉大さじ１／２ ・塩小さじ１／２弱 ・水大さじ４ レタスの細切りたっぷり\n"
#prompt_text = "以下のレシピの分類を行ってください。\n\n  [レシピ本文]\n  (title)丸鶏のローストチキン♡グレービーソースも (ingredientText)若鶏（丸鶏）1200g（1羽) ✩ハーブソルト大さじ1 ✩ローズマリー（フレッシュ）5g ✩白ワイン100cc ✩にんにく(すりおろし)1片 ✩ローリエの葉１枚 岩塩小さじ1 ブラックペッパー小さじ1/2 オリーブ油大さじ１.5 水200cc じゃがいも３個 人参２本 セロリ３本 冷凍インゲン100g グレービーソース（残った肉汁で作ります） 水200cc ★コンソメ(キューブ)1個 ★醤油大さじ2 ★砂糖大さじ１ \n"
#prompt_text = "以下のレシピの分類を行ってください。\n\n  [レシピ本文]\n  (title)カレーじゃがスパム®焼き by ホーメルフーズ (ingredientText)スパム®20%レスソルト150g じゃがいも3個 卵1個 玉ねぎ1/2個 プロセスチーズ50g カレー粉小さじ2杯 マヨネーズ20g こしょう適量 オリーブオイル適量 ケチャップお好みで \n"
#prompt_text = "以下のレシピの分類を行ってください。\n\n  [レシピ本文]\n  (title)■かんたん♪茄子とひき肉のケチャップ煮■ (ingredientText)【茄子】6～8本 【合いびき肉】250g 【★水】200ｍｌ 【★ケチャップ】大匙8 【★酒・醤油・砂糖・みりん・中農ソース】各 大匙1 【★あらびき黒胡椒】少々 【塩】適量  \n"


print("--- Generated Response ---")
print(generate_response(prompt_text))