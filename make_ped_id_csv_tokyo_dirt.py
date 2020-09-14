"""

"""
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
RACR_URL_DIR = "race_url/tokyo_dirt"
RACR_HTML_DIR = "race_html/tokyo_dirt"
PICKLE_DIR = "pickle"
CSV_DIR = "csv"
PED_CSV_PATH = CSV_DIR+"/ped_id.csv"
HORSE_PICKLE_PATH = PICKLE_DIR+"/horse_id.pickle"
PED_PICKLE_PATH = PICKLE_DIR+"/ped_id.pickle"

logger = logging.getLogger(__name__)  # ファイルの名前を渡す


def pickle_dump(obj, path):
    with open(path, mode='wb') as f:
        pickle.dump(obj, f)


def pickle_load(path):
    if not os.path.isfile(path):  # まだファイルがなければ空のset
        return set()
    with open(path, mode='rb') as f:
        data = pickle.load(f)
        return data


ped_id_columns = [
    'horse_id',
    'f_id',
    'mf_id',
    'mmf_id',
    'fmf_id'
]


def make_and_update_ped_id():

    # レースに参加した馬のidをまとめる
    horse_id_set = set()

    logger.info('making horce id set')
    for year in range(2008,  now_datetime.year+1):
        horse_race_csv = CSV_DIR+"/horse-"+str(year)+".csv"
        horse_df = pd.read_csv(horse_race_csv, sep=",")
        tmp_horse_id_set = set()
        for horse_id in horse_df['horse_id']:
            tmp_horse_id_set.add(horse_id)
        horse_id_set = horse_id_set.union(tmp_horse_id_set)
        logger.info('  made ' + str(len(tmp_horse_id_set)) + ' horce id set')
    pickle_dump(horse_id_set, HORSE_PICKLE_PATH)

    logger.info('dumped ' + str(len(horse_id_set)) + ' horce id pickle')

    # 必要な血統情報を得る
    if not os.path.isfile(PED_CSV_PATH):  # まだcsvを作成していなければ作成
        ped_id_df = pd.DataFrame(columns=ped_id_columns)
    else:
        ped_id_df = pd.read_csv(PED_CSV_PATH, sep=",")
    ped_id_set = pickle_load(PED_PICKLE_PATH)
    for num, horse_id in enumerate(horse_id_set):
        if num % 1000 == 0:
            logger.info('completed ' + str(num) + ' horses')
            ped_id_df.to_csv(PED_CSV_PATH, header=True, index=False)
            pickle_dump(ped_id_set, PED_PICKLE_PATH)
        if (ped_id_df['horse_id'] == horse_id).sum() > 0:  # すでに取得済みなら取得しない
            continue
        # 血統表のhtml を取得
        url = "https://db.netkeiba.com/horse/ped/" + str(horse_id)
        response = requests.get(url)
        response.encoding = response.apparent_encoding  # https://qiita.com/nittyan/items/d3f49a7699296a58605b
        html = response.text
        time.sleep(1)
        # 必要な血統用id を取得
        soup = BeautifulSoup(html, 'html.parser')
        try:
            furl = soup.select_one("#contents>div.db_main_deta>diary_snap>table>tr:nth-child(1)>td:nth-child(1)>a:nth-child(1)")
            if furl is not None:
                f_id = furl.get("href").split(sep='/')[2]
            else:
                f_id = None

            mfurl = soup.select_one("#contents > div.db_main_deta > diary_snap > table > tr:nth-child(17) > td:nth-child(2) > a:nth-child(1)")
            if mfurl is not None:
                mf_id = mfurl.get("href").split(sep='/')[2]
            else:
                mf_id = None

            mmfurl = soup.select_one("#contents > div.db_main_deta > diary_snap > table > tr:nth-child(25) > td:nth-child(2) > a:nth-child(1)")
            if mmfurl is not None:
                mmf_id = mmfurl.get("href").split(sep='/')[2]
            else:
                mmf_id = None

            fmfurl = soup.select_one("#contents > div.db_main_deta > diary_snap > table > tr:nth-child(9) > td:nth-child(2) > a:nth-child(1)")
            if fmfurl is not None:
                fmf_id = fmfurl.get("href").split(sep='/')[2]
            else:
                fmf_id = None

            ped_id_df = ped_id_df.append({'horse_id': horse_id,
                                          'f_id': f_id,
                                          'mf_id': mf_id,
                                          'mmf_id': mmf_id,
                                          'fmf_id': fmf_id}, ignore_index=True)
            ped_id_set.add(f_id)
            ped_id_set.add(mf_id)
            ped_id_set.add(mmf_id)
            ped_id_set.add(fmf_id)
        except Exception as e:
            logger.info(e)
            print(e)
    ped_id_df.to_csv(PED_CSV_PATH, header=True, index=False)
    pickle_dump(ped_id_set, PED_PICKLE_PATH)
    logger.info('save pedigree id csv')
    logger.info('dumped pedigree id pickle')


if __name__ == '__main__':
    formatter = "%(asctime)s [%(levelname)s]\t%(message)s"  # フォーマットを定義
    # formatter_func = "%(asctime)s\t[%(levelname)8s]\t%(message)s from %(func)" # フォーマットを定義
    logging.basicConfig(filename='logfile/'+OWN_FILE_NAME+'.logger.log', level=logging.INFO, format=formatter)

    logger.info("start making ped id set!")
    make_and_update_ped_id()
