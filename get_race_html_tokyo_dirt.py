"""
race_urlディレクトリに含まれるURLを利用して、htmlを取得する
"""
import logging
from os import path
import os
import time
from bs4 import BeautifulSoup
import requests
import datetime
import pytz
now_datetime = datetime.datetime.now(pytz.timezone('Asia/Tokyo'))


OWN_FILE_NAME = path.splitext(path.basename(__file__))[0]
RACR_URL_DIR = "race_url/tokyo_dirt"
RACR_HTML_DIR = "race_html/tokyo_dirt"


logger = logging.getLogger(__name__)  # ファイルの名前を渡す


def my_makedirs(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def get_race_html():
    # 去年までのデータ
    for year in range(2008, now_datetime.year):
        get_race_html_by_year_and_mon(year)
    # 今年のデータ
    for year in range(now_datetime.year, now_datetime.year+1):
        get_race_html_by_year_and_mon(year)


def get_race_html_by_year_and_mon(year):
    with open(RACR_URL_DIR+"/"+str(year)+".txt", "r") as f:
        save_dir = RACR_HTML_DIR+"/"+str(year)+"/"
        my_makedirs(save_dir)
        urls = f.read().splitlines()

        file_list = os.listdir(save_dir)  # get all file names

        # 取得すべき数と既に保持している数が違う場合のみ行う
        if len(urls) != len(file_list):
            logger.info("getting htmls ("+str(year) + ")")
            for url in urls:
                list = url.split("/")
                race_id = list[-2]
                save_file_path = save_dir+"/"+race_id+'.html'
                if not os.path.isfile(save_file_path):  # まだ取得していなければ取得
                    response = requests.get(url)
                    response.encoding = response.apparent_encoding  # https://qiita.com/nittyan/items/d3f49a7699296a58605b
                    html = response.text
                    time.sleep(1)
                    with open(save_file_path, 'w') as file:
                        file.write(html)
            logging.info("saved " + str(len(urls)) + " htmls ("+str(year) + ")")
        else:
            logging.info("already have " + str(len(urls)) + " htmls ("+str(year) + ")")


if __name__ == '__main__':
    formatter = "%(asctime)s [%(levelname)s]\t%(message)s"  # フォーマットを定義
    logging.basicConfig(filename='logfile/'+OWN_FILE_NAME+'.logger.log', level=logging.INFO, format=formatter)

    logger.info("start get race html!")
    get_race_html()
