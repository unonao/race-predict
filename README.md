# 競馬予測プログラム(Python)

## ファイル概要
- get_race_url.py: seleniumでレースURLを取得する
- get_race_html.py: URLを使用してhtmlを取得する
- make_csv_from_html.py: htmlからレースデータや馬のデータを抽出してCSVにする
- main.py: 以上のことを一度に実行する
- train_simple.py: 単純なモデルをkerasで作成する
- train_timesplit.py: 時系列を考慮したクロスバリデーション
- train_hyperas.py: hyperasを用いて自動パラメータチューニングを行う
- train_hyperas_no_obstacle.py: train_hyperas.pyとは違い、学習データから障害レースを取り除く

## 注意
ログはlogfileディレクトリ、htmlはrace_htmlディレクトリ、モデルはmodelディレクトリ、予想結果はpredictディレクトリに保存される。
