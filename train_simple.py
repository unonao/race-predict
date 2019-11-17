import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from io import StringIO
import sys
import requests
import traceback
import os
from os import path
OWN_FILE_NAME = path.splitext(path.basename(__file__))[0]

import logging
logger = logging.getLogger(__name__) #ファイルの名前を渡す

my_token = os.environ['LINE_TOKEN']

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss
from sklearn.metrics import confusion_matrix

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import CSVLogger

def send_line_notification(message):
    line_token = my_token
    endpoint = 'https://notify-api.line.me/api/notify'
    message = "\n{}".format(message)
    payload = {'message': message}
    headers = {'Authorization': 'Bearer {}'.format(line_token)}
    requests.post(endpoint, data=payload, headers=headers)


def train_test_time_split(dataflame, train_ratio=0.8):
    """
    時系列を加味してデータをsplit
    """
    X =dataflame.sort_values("date")
    train_size = int(len(X) * train_ratio)

    logger.info("split trian and test :{} (train_ratio:{})".format(X["date"][train_size] , train_ratio))

    return X[0:train_size].copy().reset_index(drop=True), X[train_size:len(X)].copy().reset_index(drop=True)

def label_split_and_drop(X_df, target_name):
    """
    target_nameをYに分割して、Xから余分なカラムを削除し、numpyの形式にする
    """
    Y = X_df[target_name].values
    X = X_df.drop(['is_tansyo','is_hukusyo','date','race_id'], axis=1).values
    #logger.info("train columns: {}".format(X_df.drop(['is_tansyo','is_hukusyo','date','race_id' ], axis=1).columns))
    sc = StandardScaler()
    X = sc.fit_transform(X)
    return X, Y


def build_model(df_columns_len):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(300, kernel_regularizer=tf.keras.regularizers.l2(0.001), activation=tf.nn.relu, input_dim=df_columns_len),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(100, kernel_regularizer=tf.keras.regularizers.l2(0.001), activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation=tf.nn.sigmoid) # 出力は一つ。シグモイド関数
    ])

    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(),
        metrics=['accuracy'])

    return model

def plot_history(history,name):
    fig, (axL, axR) = plt.subplots(ncols=2, figsize=(10,4))
    # for loss
    axL.plot(history.history['loss'],label="loss for training")
    axL.plot(history.history['val_loss'],label="loss for validation")
    axL.set_title('model loss')
    axL.set_xlabel('epoch')
    axL.set_ylabel('loss')
    axL.legend(loc='upper right')
    axR.plot(history.history['accuracy'],label="acc for training")
    axR.plot(history.history['val_accuracy'],label="acc for validation")
    axR.set_title('model accuracy')
    axR.set_xlabel('epoch')
    axR.set_ylabel('accuracy')
    axR.legend(loc='upper right')
    # figureの保存
    plt.savefig("model/{}_{}_model_plot_history.png".format(OWN_FILE_NAME,name))


def train_model(train_data,train_label,val_data,val_label,target_name,):
    logger.info("start train!"))
    callbacks = []
    callbacks.append(EarlyStopping(monitor='val_loss', patience=3))
    callbacks.append(CSVLogger("model/{}_{}_history.csv".format(OWN_FILE_NAME,target_name)))
    model=build_model(train_data.shape[1])
    history = model.fit(train_data,
        train_label,
        validation_data=(val_data, val_label),
        epochs=3,
        batch_size=256,
        callbacks=callbacks)
    # モデルの保存
    model.save("model/{}_{}_model.h5".format(OWN_FILE_NAME,target_name))

    logger.info("{} loss:\t{}".format(target_name, history.history["loss"]))
    logger.info("{} val_loss:\t{}".format(target_name, history.history["val_loss"]))
    # 可視化
    #plot_history(history, target_name)
    return model

def keras_train(target_name):
    logger.info("start train for {}".format(target_name))

    final_df = pd.read_csv("csv/final_data.csv", sep=",")
    train_df, test_df = train_test_time_split(final_df)
    X_train, Y_train = label_split_and_drop(train_df, target_name)
    X_test, Y_test = label_split_and_drop(test_df, target_name)

    predict_proba_results = np.zeros(len(Y_test))
    predict_proba_results = predict_proba_results.reshape(len(Y_test),1)

    train_size = int(len(Y_train) * 0.8)
    train_data = X_train[0:train_size]
    train_label = Y_train[0:train_size]
    val_data = X_train[train_size:len(Y_train)]
    val_label = Y_train[train_size:len(Y_train)]
    model = train_model(train_data,train_label,val_data,val_label,target_name)
    predict_proba_results = model.predict_proba(X_test)

    with StringIO() as buf:
            # StringIOに書き込む
            model.summary(print_fn=lambda x: buf.write(x + "\n"))
            # StringIOから取得
            text = buf.getvalue()
    logger.info("model summary:\n{}".format(text))


    predict_proba_results = predict_proba_results.flatten()

    # test データでのloglossを確認
    logger.info("{} test_log_loss:\t\t{}".format(target_name, log_loss(Y_test,  predict_proba_results)))

    # 混同行列
    predict_results = np.where(predict_proba_results > 0.5, 1, 0) # 確率に応じて0,1に変換
    logger.info("{} confusion_matrix:\n{}\n".format(target_name, confusion_matrix(Y_test, predict_results)))

    # 結果の保存のためにシリーズにする
    return pd.Series(data=predict_proba_results, name="predict_{}".format(target_name), dtype='float')


if __name__ == '__main__':
    try:
        formatter_func = "%(asctime)s - %(module)s.%(funcName)s [%(levelname)s]\t%(message)s" # フォーマットを定義
        logging.basicConfig(filename='logfile/'+OWN_FILE_NAME+'.logger.log', level=logging.INFO, format=formatter_func)

        is_tansyo_se = keras_train('is_tansyo')
        is_hukusyo_se = keras_train('is_hukusyo')

        # 結果の保存
        final_df = pd.read_csv("csv/final_data.csv", sep=",")
        _, test_df = train_test_time_split(final_df)
        predicted_test_df = pd.concat([test_df, is_tansyo_se,is_hukusyo_se], axis=1)
        predicted_test_df.to_csv("predict/{}_best_predicted_test.csv".format(OWN_FILE_NAME), index=False)

        send_line_notification(OWN_FILE_NAME+" end!")
    except Exception as e:
        t, v, tb = sys.exc_info()
        for str in traceback.format_exception(t,v,tb):
            str = "\n"+str
            logger.error(str)
            send_line_notification(str)
