# AutoRegressive-VITS

(WIP) オートレグレッシブ トランスフォーマーとVITSを使用したテキストから音声への変換
## 注意
+ モデルの効果は完全には検証されておらず、必ずしも良い結果が出るとは限りません。注意して進めてください。事前訓練されたモデルはまだ訓練中です。
+ ゼロからの訓練には膨大なデータが必要です（少なくとも数千時間？）（valle、speartts、soundstormに類似）。データが少ないと必ずしも良い結果は得られません。
+ VITS+リファレンスにはzeroshot方向で大きな制限があるため、このリポジトリはzeroshotを目指していません。このリポジトリの目標は、大規模な言語モデルの事前訓練がある場合に、オートレグレッシブ言語モデルの力を借りて、小規模なデータでのファインチューニング後に良い韻律を得ることです。
+ 初期のいくつかの [合成サンプル](https://huggingface.co/innnky/ar-tts-models/tree/main/gpt-vits) を簡単に更新しました。
## Todo
+ [x] 原神データでの訓練
+ [x] さらに多くの中国語オープンソースデータを収集して訓練（約600時間を予定）し、事前訓練モデルを公開（x）--> 分布外のテキストの性能が非常に悪く、例えば古典中国語の読み上げや長文の効果が悪く、異常な動作をすることがあります。
  + [ ] 分布外の性能を改善するために、単語レベルのBERTを追加し、音素レベルまで繰り返す。
  + [ ] 同じ話者の複数のデータを1つのオーディオファイルに結合して平均データ長を延ばし、長文合成の安定性を向上させる。
  + [ ] 長文合成の安定性を向上させるために、RoPE相対位置エンコーディングに変更？
+ [ ] ファインチューニング関連のコードを作成し、話者IDのサポートを追加。
+ [ ] 日本語と英語のテキストフロントエンドを最適化し、さらに多くの日本語と英語データを収集（各言語約600時間を予定）して訓練し、事前訓練モデルを公開。

## 構造
![structure.png]([resources%2Fstructure.png](https://github.com/innnky/ar-vits/blob/master/resources/structure.png?raw=true))

+ [SoundStorm](https://github.com/yangdongchao/SoundStorm/tree/master/soundstorm/s1/AR)からのデコーダーオンリーテキスト2セマンティック
+ [VITS](https://github.com/jaywalnut310/vits)からのVITS
+ [TransferTTS](https://github.com/hcy71o/TransferTTS)からのリファレンスエンコーダー

## 訓練パイプライン
1. S2 VITSデコーダーと量子化器を共同で訓練
2. セマンティックトークンを抽出
3. S1 テキストをセマンティックに訓練

## VITS S2 訓練
+ resample.py
+ gen_phonemes.py
+ extract_ssl_s2.py
+ gen_filelist_s2.py
+ train_s2.py

## GPT S1 訓練
+ extract_vq_s1.py
+ gen_filelist_s1.py
+ train_s1.py

## 推論
+ s1_infer.py/s2_infer.py （進行中）

## 事前訓練されたモデル
+ 進行中
