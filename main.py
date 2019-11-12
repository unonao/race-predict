from get_race_url import get_race_url
from get_race_html import get_race_html
from make_csv_from_html import make_csv_from_html

import sys
import traceback
import requests
import os
from os import path
OWN_FILE_NAME = path.splitext(path.basename(__file__))[0]

import logging

logger = logging.getLogger(__name__) #ファイルの名前を渡す

my_token = os.environ['LINE_TOKEN']

def send_line_notification(message):
    line_token = my_token
    endpoint = 'https://notify-api.line.me/api/notify'
    message = "\n{}".format(message)
    payload = {'message': message}
    headers = {'Authorization': 'Bearer {}'.format(line_token)}
    requests.post(endpoint, data=payload, headers=headers)

def update():
    get_race_url()
    get_race_html()
    make_csv_from_html()


if __name__ == '__main__':
    try:
        formatter_func = "%(asctime)s - %(module)s.%(funcName)s [%(levelname)s]\t%(message)s" # フォーマットを定義
        logging.basicConfig(filename='logfile/'+OWN_FILE_NAME+'.logger.log', level=logging.INFO, format=formatter_func)
        logger.info("start updating!")
        update()
        send_line_notification(OWN_FILE_NAME+" end!")
    except Exception as e:
        t, v, tb = sys.exc_info()
        for str in traceback.format_exception(t,v,tb):
            str = "\n"+str
            logger.error(str)
            send_line_notification(str)
