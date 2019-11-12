"""
race_urlに含まれるURLを利用して、htmlを取得する
"""
import requests
from bs4 import BeautifulSoup

import time

import os
from os import path
OWN_FILE_NAME = path.splitext(path.basename(__file__))[0]
RACR_URL_DIR = "race_url"
RACR_HTML_DIR = "race_html"


import logging
formatter = "%(asctime)s [%(levelname)s]\t%(message)s" # フォーマットを定義
#formatter_func = "%(asctime)s\t[%(levelname)8s]\t%(message)s from %(func)" # フォーマットを定義
logging.basicConfig(filename='logfile/'+OWN_FILE_NAME+'.logger.log', level=logging.INFO, format=formatter)
logger = logging.getLogger(__name__) #ファイルの名前を渡す
logger.info("start script...")


def my_makedirs(path):
    if not os.path.isdir(path):
        os.makedirs(path)

def get_race_html():
    for year in range(2008, 2020):
        for month in range(1, 13):
                get_race_html_by_year_and_mon(year,month)

def get_race_html_by_year_and_mon(year,month):
    with open(RACR_URL_DIR+"/"+str(year)+"-"+str(month)+".txt", "r") as f:
        save_dir = RACR_HTML_DIR+"/"+str(year)+"/"+str(month)
        my_makedirs(save_dir)
        urls = f.read().splitlines()
        logger.info("getting htmls ("+str(year) +" "+ str(month) + ")")
        for url in urls:
            list = url.split("/")
            race_id = list[-2]
            save_file_path = save_dir+"/"+race_id+'.html'
            if not os.path.isfile(save_file_path): # まだ取得していなければ取得
                response = requests.get(url)
                response.encoding = response.apparent_encoding # https://qiita.com/nittyan/items/d3f49a7699296a58605b
                html = response.text
                time.sleep(1)
                with open(save_file_path, 'w') as file:
                    file.write(html)
    logging.info("saved " + str(len(urls)) +" htmls ("+str(year) +" "+ str(month) + ")")

if __name__ == '__main__':
    print("start get race html!")
    get_race_html()
