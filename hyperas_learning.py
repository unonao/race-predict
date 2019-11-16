"""
"""
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

import keras
from keras.callbacks import EarlyStopping
from keras.callbacks import CSVLogger
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation

from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform


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
    X = X_df.drop(['is_tansyo','is_hukusyo','date','race_id' ], axis=1).values
    #logger.info("train columns: {}".format(X_df.drop(['is_tansyo','is_hukusyo','date','race_id' ], axis=1).columns))
    sc = StandardScaler()
    X = sc.fit_transform(X)
    return X, Y


def create_model(X_train, Y_train, X_test, Y_test):
    train_size = int(len(Y_train) * 0.8)
    train_data = X_train[0:train_size]
    train_label = Y_train[0:train_size]
    val_data = X_train[train_size:len(Y_train)]
    val_label = Y_train[train_size:len(Y_train)]

    callbacks = []
    callbacks.append(EarlyStopping(monitor='val_loss', patience=3))

    model = Sequential()
    model.add(Dense({{choice([256, 512,1024])}}, kernel_regularizer=keras.regularizers.l2(0.001), activation="relu", input_dim=train_data.shape[1]))
    model.add(Dropout({{uniform(0, 0.5)}}))
    model.add(Dense({{choice([64, 128, 256])}}, kernel_regularizer=keras.regularizers.l2(0.001), activation="relu"))
    model.add(Dropout({{uniform(0, 0.5)}}))

    if {{choice(['three', 'four'])}} == 'three':
        pass
    elif {{choice(['three', 'four'])}} == 'four':
        model.add(Dense(16, kernel_regularizer=keras.regularizers.l2(0.001), activation="relu"))
        model.add(Dropout({{uniform(0, 0.5)}}))

    model.add(Dense(1, activation="sigmoid"))

    model.compile(
        loss='binary_crossentropy',
        optimizer=keras.optimizers.Adam(),
        metrics=['accuracy'])

    history = model.fit(train_data,
        train_label,
        validation_data=(val_data, val_label),
        epochs=30,
        batch_size=64,
        callbacks=callbacks)

    val_loss, val_acc = model.evaluate(X_test, Y_test, verbose=0)
    print('Best validation loss of epoch:', val_loss)
    return {'loss': val_loss, 'status': STATUS_OK, 'model': model}

def prepare_data_is_tansyo():
    target_name='is_tansyo'
    final_df = pd.read_csv("csv/final_data.csv", sep=",")

    train_ratio = 0.8
    X =final_df.sort_values("date")
    train_size = int(len(X) * train_ratio)
    train_df = X[0:train_size].copy().reset_index(drop=True)
    test_df = X[train_size:len(X)].copy().reset_index(drop=True)

    Y_train = train_df[target_name].values
    X_train = train_df.drop(['is_tansyo','is_hukusyo','date','race_id' ], axis=1).values
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)

    Y_test = test_df[target_name].values
    X_test = test_df.drop(['is_tansyo','is_hukusyo','date','race_id' ], axis=1).values
    sc = StandardScaler()
    X_test = sc.fit_transform(X_test)

    return X_train, Y_train, X_test, Y_test

"""


"""

def prepare_data_is_hukusyo():
    target_name='is_tansyo'
    final_df = pd.read_csv("csv/final_data.csv", sep=",")

    train_ratio = 0.8
    X =final_df.sort_values("date")
    train_size = int(len(X) * train_ratio)
    train_df = X[0:train_size].copy().reset_index(drop=True)
    test_df = X[train_size:len(X)].copy().reset_index(drop=True)

    Y_train = train_df[target_name].values
    X_train = train_df.drop(['is_tansyo','is_hukusyo','date','race_id' ], axis=1).values
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)

    Y_test = test_df[target_name].values
    X_test = test_df.drop(['is_tansyo','is_hukusyo','date','race_id' ], axis=1).values
    sc = StandardScaler()
    X_test = sc.fit_transform(X_test)

    return X_train, Y_train, X_test, Y_test

def hyperas_learn(target_name):
    """
    最適なモデルの作成・保存・結果のリターン
    """
    logger.info("start train for {}".format(target_name))
    if target_name=='is_tansyo':
        best_run, best_model = optim.minimize(model=create_model,
                                              data=prepare_data_is_tansyo,
                                              algo=tpe.suggest,
                                              max_evals=15,
                                              trials=Trials())
        _, _, X_test, Y_test = prepare_data_is_tansyo()
    elif target_name=='is_hukusyo':
        best_run, best_model = optim.minimize(model=create_model,
                                              data=prepare_data_is_hukusyo,
                                              algo=tpe.suggest,
                                              max_evals=15,
                                              trials=Trials())
        _, _, X_test, Y_test = prepare_data_is_hukusyo()

    best_model.save("model/best_{}_model.h5".format(target_name))

    with StringIO() as buf:
            # StringIOに書き込む
            best_model.summary(print_fn=lambda x: buf.write(x + "\n"))
            # StringIOから取得
            text = buf.getvalue()
    logger.info("best_model summary:\n{}".format(text))
    logger.info("best_run:\t{}".format(best_run))


    val_loss, val_acc = best_model.evaluate(X_test, Y_test)
    logger.info("test loss:\t{}".format(val_loss))
    logger.info("test acc:\t{}".format(val_acc))

    predict_proba_results = best_model.predict_proba(X_test)

    # 混同行列
    predict_results = np.where(predict_proba_results > 0.5, 1, 0) # 確率に応じて0,1に変換
    logger.info("{} confusion_matrix:\n{}\n".format(target_name, confusion_matrix(Y_test, predict_results)))

    # 結果の保存のためにシリーズにする
    predict_proba_results = predict_proba_results.flatten()
    return pd.Series(data=predict_proba_results, name="predict_{}".format(target_name), dtype='float')



if __name__ == '__main__':
    try:
        formatter_func = "%(asctime)s - %(module)s.%(funcName)s [%(levelname)s]\t%(message)s" # フォーマットを定義
        logging.basicConfig(filename='logfile/'+OWN_FILE_NAME+'.logger.log', level=logging.INFO, format=formatter_func)

        is_tansyo_se = hyperas_learn('is_tansyo')
        is_hukusyo_se = hyperas_learn('is_hukusyo')

        # 結果の保存
        final_df = pd.read_csv("csv/final_data.csv", sep=",")
        _, test_df = train_test_time_split(final_df)
        predicted_test_df = pd.concat([test_df, is_tansyo_se,is_hukusyo_se], axis=1)
        predicted_test_df.to_csv("predict/best_predicted_test.csv", index=False)

        send_line_notification(OWN_FILE_NAME+" end!")
    except Exception as e:
        t, v, tb = sys.exc_info()
        for str in traceback.format_exception(t,v,tb):
            str = "\n"+str
            logger.error(str)
            send_line_notification(str)
