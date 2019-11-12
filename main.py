from get_race_url import get_race_url
from get_race_html import get_race_html
from make_csv_from_html import make_csv_from_html

import os
from os import path
OWN_FILE_NAME = path.splitext(path.basename(__file__))[0]

import logging

logger = logging.getLogger(__name__) #ファイルの名前を渡す

def update():
    get_race_url()
    get_race_html()
    make_csv_from_html()


if __name__ == '__main__':
    formatter_func = "%(asctime)s - %(module)s.%(funcName)s [%(levelname)s]\t%(message)s" # フォーマットを定義
    logging.basicConfig(filename='logfile/'+OWN_FILE_NAME+'.logger.log', level=logging.INFO, format=formatter_func)

    logger.info("start updating!")
    update()
