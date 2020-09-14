"""
ped_id_html からデータ抽出して産駒csvを作成する
"""
import logging
from os import path
import os
import re
import time
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import datetime
import pytz
import pickle
now_datetime = datetime.datetime.now(pytz.timezone('Asia/Tokyo'))


OWN_FILE_NAME = path.splitext(path.basename(__file__))[0]
SIRE_CSV = "csv/sire.csv"
HTML_DIR = "sire_html"

logger = logging.getLogger(__name__)  # ファイルの名前を渡す


sire_columns = [
    'ped_id',
    'rank',
    'win_horse_rate',
    'win_times_rate',
    'win_jusyo_rate',
    'win_special_rate',
    'win_plane_rate',
    'win_turf_rate',
    'win_dirt_rate',
    'EI'
]


def make_sire_csv():
    if not os.path.isfile(SIRE_CSV):  # まだcsvがなければ生成
        sire_df = pd.DataFrame(columns=sire_columns)
    else:
        sire_df = pd.read_csv(SIRE_CSV, sep=",")

    if os.path.isdir(HTML_DIR):
        file_list = os.listdir(HTML_DIR)  # get all file names
        logger.info(" appending " + str(len(file_list)) + " datas to csv")
        for file_name in file_list:
            with open(HTML_DIR+"/"+file_name, "r") as f:
                html = f.read()
                list = file_name.split(".")
                ped_id = list[-2]
                if (sire_df['ped_id'] == ped_id).sum() > 0:
                    continue
                sire_list = get_sire_data_by_html(ped_id, html)
                sire_se = pd.Series(sire_list, index=sire_df.columns)
                sire_df = sire_df.append(sire_se, ignore_index=True)
        sire_df.to_csv(SIRE_CSV, header=True, index=False)
        logger.info(' (rows, columns) of sire_df:\t' + str(sire_df.shape))
    else:
        logger.info("could not found " + str(HTML_DIR))


def get_sire_data_by_html(ped_id, html):
    sire_list = [ped_id]  # 'ped_id'
    soup = BeautifulSoup(html, 'html.parser')

    # 'rank'
    rank_tag = soup.select_one("#contents > div.db_main_deta > table > tr:nth-child(3) > td:nth-child(2)")
    if rank_tag is not None:
        rank = int(rank_tag.get_text().strip("\n"))
    else:
        rank = None
    sire_list.append(rank)

    #  'win_horse_rate'
    horse_num_tag = soup.select_one("#contents > div.db_main_deta > table > tr:nth-child(3) > td:nth-child(3)")
    win_horse_num_tag = soup.select_one("#contents > div.db_main_deta > table > tr:nth-child(3) > td:nth-child(4)")
    if horse_num_tag is not None and win_horse_num_tag is not None:
        horse_num = float(horse_num_tag.get_text().strip("\n").replace(',', ''))
        win_horse_num = float(win_horse_num_tag.get_text().strip("\n").replace(',', ''))
        win_horse_rate = win_horse_num/(horse_num+1e-10)
    else:
        horse_num = None
        win_horse_num = None
        win_horse_rate = None
    sire_list.append(win_horse_rate)

    # 'win_times_rate'
    join_num_tag = soup.select_one("#contents > div.db_main_deta > table > tr:nth-child(3) > td:nth-child(5)")
    win_num_tag = soup.select_one("#contents > div.db_main_deta > table > tr:nth-child(3) > td:nth-child(6)")
    if join_num_tag is not None and win_num_tag is not None:
        join_num = float(join_num_tag.get_text().strip("\n").replace(',', ''))
        win_num = float(win_num_tag.get_text().strip("\n").replace(',', ''))
        win_times_rate = win_num/(join_num+1e-10)
    else:
        join_num = None
        win_num = None
        win_times_rate = None
    sire_list.append(win_times_rate)

    # 'win_jusyo_rate'
    join_jusyo_tag = soup.select_one("#contents > div.db_main_deta > table > tr:nth-child(3) > td:nth-child(7)")
    win_jusyo_tag = soup.select_one("#contents > div.db_main_deta > table > tr:nth-child(3) > td:nth-child(8)")
    if join_jusyo_tag is not None and win_jusyo_tag is not None:
        join_jusyo = float(join_jusyo_tag.get_text().strip("\n").replace(',', ''))
        win_jusyo = float(win_jusyo_tag.get_text().strip("\n").replace(',', ''))
        win_jusyo_rate = win_jusyo/(join_jusyo+1e-10)
    else:
        join_jusyo = None
        win_jusyo = None
        win_jusyo_rate = None
    sire_list.append(win_jusyo_rate)

    # 'win_special_rate'
    join_special_tag = soup.select_one("#contents > div.db_main_deta > table > tr:nth-child(3) > td:nth-child(9)")
    win_special_tag = soup.select_one("#contents > div.db_main_deta > table > tr:nth-child(3) > td:nth-child(10)")
    if join_special_tag is not None and win_special_tag is not None:
        join_special = float(join_special_tag.get_text().strip("\n").replace(',', ''))
        win_special = float(win_special_tag.get_text().strip("\n").replace(',', ''))
        win_special_rate = win_special/(join_special+1e-10)
    else:
        join_special = None
        win_special = None
        win_special_rate = None
    sire_list.append(win_special_rate)

    # 'win_plane_rate'
    join_plane_tag = soup.select_one("#contents > div.db_main_deta > table > tr:nth-child(3) > td:nth-child(11)")
    win_plane_tag = soup.select_one("#contents > div.db_main_deta > table > tr:nth-child(3) > td:nth-child(12)")
    if join_plane_tag is not None and win_plane_tag is not None:
        join_plane = float(join_plane_tag.get_text().strip("\n").replace(',', ''))
        win_plane = float(win_plane_tag.get_text().strip("\n").replace(',', ''))
        win_plane_rate = win_plane/(join_plane+1e-10)
    else:
        join_plane = None
        win_plane = None
        win_plane_rate = None
    sire_list.append(win_plane_rate)

    # 'win_turf_rate'
    join_turf_tag = soup.select_one("#contents > div.db_main_deta > table > tr:nth-child(3) > td:nth-child(13)")
    win_turf_tag = soup.select_one("#contents > div.db_main_deta > table > tr:nth-child(3) > td:nth-child(14)")
    if join_turf_tag is not None and win_turf_tag is not None:
        join_turf = float(join_turf_tag.get_text().strip("\n").replace(',', ''))
        win_turf = float(win_turf_tag.get_text().strip("\n").replace(',', ''))
        win_turf_rate = win_turf/(join_turf+1e-10)
    else:
        join_turf = None
        win_turf = None
        win_turf_rate = None
    sire_list.append(win_turf_rate)

    # 'win_dirt_rate'
    join_dirt_tag = soup.select_one("#contents > div.db_main_deta > table > tr:nth-child(3) > td:nth-child(15)")
    win_dirt_tag = soup.select_one("#contents > div.db_main_deta > table > tr:nth-child(3) > td:nth-child(16)")
    if join_dirt_tag is not None and win_dirt_tag is not None:
        join_dirt = float(join_dirt_tag.get_text().strip("\n").replace(',', ''))
        win_dirt = float(win_dirt_tag.get_text().strip("\n").replace(',', ''))
        win_dirt_rate = win_dirt/(join_dirt+1e-10)
    else:
        join_dirt = None
        win_dirt = None
        win_dirt_rate = None
    sire_list.append(win_dirt_rate)

    # 'EI'
    ei_tag = soup.select_one("#contents > div.db_main_deta > table > tr:nth-child(3) > td:nth-child(18)")
    if ei_tag is not None:
        ei = float(ei_tag.get_text().strip("\n").replace(',', ''))
    else:
        ei = None
    sire_list.append(ei)

    return sire_list


if __name__ == '__main__':
    formatter = "%(asctime)s [%(levelname)s]\t%(message)s"  # フォーマットを定義
    # formatter_func = "%(asctime)s\t[%(levelname)8s]\t%(message)s from %(func)" # フォーマットを定義
    logging.basicConfig(filename='logfile/'+OWN_FILE_NAME+'.logger.log', level=logging.INFO, format=formatter)

    logger.info("start making csv!")
    make_sire_csv()

    # test
    '''
    file_name = '000a0010f3.html'
    with open(HTML_DIR+"/"+file_name, "r") as f:
        html = f.read()
        list = file_name.split(".")
        ped_id = list[-2]
        sire_list = get_sire_data_by_html(ped_id, html)
    print(sire_list)
    '''
