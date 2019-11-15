import numpy as np
import pandas as pd

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

import tensorflow as tf

def send_line_notification(message):
    line_token = my_token
    endpoint = 'https://notify-api.line.me/api/notify'
    message = "\n{}".format(message)
    payload = {'message': message}
    headers = {'Authorization': 'Bearer {}'.format(line_token)}
    requests.post(endpoint, data=payload, headers=headers)


def train_test_time_split(dataflame, train_ratio=0.8):
    X =dataflame.sort_values("date")
    train_size = int(len(X) * train_ratio)

    logger.info("split trian and test :{} (train_ratio:{})".format(X["date"][train_size] , train_ratio))

    return X[0:train_size], X[train_size:len(X)]

def label_split_and_drop(X_df, target):
    # targetをYに分割して、Xから余分なカラムを削除し、numpyの形式にする
    Y = X_df[target].values
    X = X_df.drop([target], axis=1).drop(['date'], axis=1).drop(['race_id'], axis=1).values
    return X, Y

def build_model(df_columns_len):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(300, kernel_regularizer=tf.keras.regularizers.l2(0.001), activation=tf.nn.relu, input_dim=df_columns_len),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(300, kernel_regularizer=tf.keras.regularizers.l2(0.001), activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation=tf.nn.sigmoid) # 出力は一つ。シグモイド関数。ソフトマックスで確率を出すのは...？
    ])

    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(),
        metrics=['accuracy'])

    return model

def compare_TV(history,number):
    import matplotlib.pyplot as plt
    # Setting Parameters
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    # 1) Accracy Plt
    plt.plot(epochs, acc, 'bo' ,label = 'training acc')
    plt.plot(epochs, val_acc, 'b' , label= 'validation acc')
    plt.title('Training and Validation acc')
    plt.legend()
    plt.figure()
    # 2) Loss Plt
    plt.plot(epochs, loss, 'bo' ,label = 'training loss')
    plt.plot(epochs, val_loss, 'b' , label= 'validation loss')
    plt.title('Training and Validation loss')
    plt.legend()

    plt.show()
    # figureの保存
    plt.savefig("png/plot{}.png".format(number))


def keras_train():
    final_df = pd.read_csv("csv/final_data.csv", sep=",")
    train, test = train_test_time_split(final_df)
    X_train, Y_train = label_split_and_drop(train, 'is_tansyo')
    X_test, Y_test = label_split_and_drop(test, 'is_tansyo')

    # 時系列データで有ることを考慮してCVを切る
    tscv = TimeSeriesSplit(n_splits=5)
    number = 0
    all_loss=[]
    all_val_loss=[]
    all_acc=[]
    all_val_acc=[]
    for train_index, val_index in tscv.split(X_train,Y_train):
        logger.info("start !")
        train_data=X_train[train_index]
        train_label=Y_train[train_index]
        val_data=X_train[val_index]
        val_label=Y_train[val_index]

        model=build_model()

        history = model.fit(train_data,
            train_label,
            validation_data=(val_data, val_label),
            epochs=30,
            batch_size=32)

        all_loss.append(history.history['loss'])
        all_val_loss.append(history.history['val_loss'])
        all_acc.append(history.history['acc'])
        all_val_acc.append(history.history['val_acc'])

        # 可視化
        compare_TV(history, number)
        number += 1


    ave_all_loss=[
        np.mean([x[i] for x in all_loss]) for i in range(ep)]
    ave_all_val_loss=[
        np.mean([x[i] for x in all_val_loss]) for i in range(ep)]
    ave_all_acc=[
        np.mean([x[i] for x in all_acc]) for i in range(ep)]
    ave_all_val_acc=[
        np.mean([x[i] for x in all_val_acc]) for i in range(ep)]

    logger.info("\n    ave_all_loss:{}".format(ave_all_loss))
    logger.info("\n    ave_all_val_loss:{}".format(ave_all_val_loss))
    logger.info("\n    ave_all_acc:{}".format(ave_all_acc))
    logger.info("\n    ave_all_val_acc:{}".format(ave_all_val_acc))

    logger.info("\n    all_loss:{}".format(all_loss))
    logger.info("\n    all_val_loss:{}".format(all_val_loss))
    logger.info("\n    all_acc:{}".format(all_acc))
    logger.info("\n    all_val_acc:{}".format(all_val_acc))

if __name__ == '__main__':
    try:
        formatter_func = "%(asctime)s - %(module)s.%(funcName)s [%(levelname)s]\t%(message)s" # フォーマットを定義
        logging.basicConfig(filename='logfile/'+OWN_FILE_NAME+'.logger.log', level=logging.INFO, format=formatter_func)
        logger.info("start train!")
        keras_train()
        send_line_notification(OWN_FILE_NAME+" end!")
    except Exception as e:
        t, v, tb = sys.exc_info()
        for str in traceback.format_exception(t,v,tb):
            str = "\n"+str
            logger.error(str)
            send_line_notification(str)
