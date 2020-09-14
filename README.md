# 競馬予測プログラム(Python)

## 流れ
1. 対象レースのURLを取得。東京でダートのみ (get_race_url)
2. URLからレースのhtmlソースを取得 (get_race_html)
3. レースごとに必要な情報をあつめる (make_race_csv)
    - レース概要
    - 馬ごとの成績
4. 参加した馬をさかのぼって血統のidを取得し、辞書やcsvで保存(make_ped_id_csv)
5. 血統のid から馬の産駒htmlの取得(get_sire_html)
6. 血統のid から馬の産駒csvの作成（make_sire_csv)
7. データ整形
8. 馬csv,レースcsv,産駒csv を結合
9. lightgbm で学習
10. できたモデルで精度確認


## 概要
- race_url: 対象レースのURLのみのテキストデータ
- race_html: URL から取得したhtmlファイル
- horse_ped_html: レースに参加した馬の血統表html
- csv
    - race: レース情報まとめ
    - horse: 対象レースでの成績まとめ
    - horse_ped_id: 対象レースの馬の先祖のid
    - horse_sire: 上記のidをもとに手に入れた


- get_race_url.py: seleniumでレースURLを取得する
- get_race_html.py: URLを使用してhtmlを取得する
- make_csv_from_html.py: htmlからレースデータや馬のデータを抽出してCSVにする
- main.py: 以上のことを一度に実行する
- data_clean.ipynb: 得られたcsvを整形する
- train_simple.py: 単純なモデルをkerasで作成する
- train_timesplit.py: 時系列を考慮したクロスバリデーション
- train_hyperas.py: hyperasを用いて自動パラメータチューニングを行う
- train_hyperas_no_obstacle.py: train_hyperas.pyとは違い、学習データから障害レースを取り除く
- evaluate_prediction.ipynb: モデルの予測値の評価を行う

## 注意
ログはlogfileディレクトリ、htmlはrace_htmlディレクトリ、モデルはmodelディレクトリ、予想結果はpredictディレクトリに保存される。


## 血統情報などの追加
https://db.netkeiba.com/horse/sire/”馬番号”/
で産駒成績を得られる
父＞母父＞母母父＞父母父？

## 対象レースの制限
競馬場を東京のみに
コースをダートのみに変更
（ダートは芝よりもコースが安定していてレースが荒れにくいため予想しやすい）
