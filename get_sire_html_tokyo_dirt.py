"""
race_htmlに含まれるhtmlを利用して、データを生成する
"""
import math
import pickle
import logging
from os import path
import os
import re
import time
import pandas as pd
import numpy as np
import requests
import lxml.html
from bs4 import BeautifulSoup
import datetime
import pytz
now_datetime = datetime.datetime.now(pytz.timezone('Asia/Tokyo'))


OWN_FILE_NAME = path.splitext(path.basename(__file__))[0]
PICKLE_DIR = "pickle"
SAVE_DIR = "sire_html"
PED_PICKLE_PATH = PICKLE_DIR+"/ped_id.pickle"

logger = logging.getLogger(__name__)  # ファイルの名前を渡す


def pickle_dump(obj, path):
    with open(path, mode='wb') as f:
        pickle.dump(obj, f)


def pickle_load(path):
    if not os.path.isfile(path):  # まだファイルがなければ空のset
        return set()
    else:
        with open(path, mode='rb') as f:
            data = pickle.load(f)
            return data


def get_ped_id_html():
    ped_id_set = pickle_load(PED_PICKLE_PATH)

    logger.info("getting htmls ("+str(len(ped_id_set))+")")
    for num, ped_id in enumerate(ped_id_set):
        if type(ped_id) == float and math.isnan(ped_id):
            continue
        if num % 100 == 0:
            logger.info("  get " + str(num))
        save_file_path = SAVE_DIR+"/"+str(ped_id)+'.html'
        url = "https://db.netkeiba.com/horse/sire/" + str(ped_id)
        if not os.path.isfile(save_file_path):  # まだ取得していなければ取得
            response = requests.get(url)
            response.encoding = response.apparent_encoding  # https://qiita.com/nittyan/items/d3f49a7699296a58605b
            html = response.text
            time.sleep(1)
            with open(save_file_path, 'w') as file:
                file.write(html)
    logging.info("saved " + str(len(ped_id_set)) + " htmls")


if __name__ == '__main__':
    formatter = "%(asctime)s [%(levelname)s]\t%(message)s"  # フォーマットを定義
    # formatter_func = "%(asctime)s\t[%(levelname)8s]\t%(message)s from %(func)" # フォーマットを定義
    logging.basicConfig(filename='logfile/'+OWN_FILE_NAME+'.logger.log', level=logging.INFO, format=formatter)

    logger.info("start getting ped id html!")
    get_ped_id_html()
