"""
race_htmlに含まれるhtmlを利用して、データを生成する
"""
import datetime
import pytz
now_datetime = datetime.datetime.now(pytz.timezone('Asia/Tokyo'))

from bs4 import BeautifulSoup
import numpy as np
import pandas as pd

import time
import re
import os
from os import path
OWN_FILE_NAME = path.splitext(path.basename(__file__))[0]
RACR_URL_DIR = "race_url"
RACR_HTML_DIR = "race_html"
CSV_DIR = "csv"

import logging
logger = logging.getLogger(__name__) #ファイルの名前を渡す


race_data_columns=[
    'race_id',
    'race_round',
    'race_title',
    'race_course',
    'weather',
    'ground_status',
    'time',
    'date',
    'where_racecourse',
    'total_horse_number',
    'frame_number_first',
    'horse_number_first',
    'frame_number_second',
    'horse_number_second',
    'frame_number_third',
    'horse_number_third',
    'tansyo',
    'hukuren_first',
    'hukuren_second',
    'hukuren_third',
    'renhuku3',
    'rentan3'
    ]

horse_data_columns=[
    'race_id',
    'rank',
    'frame_number',
    'horse_number',
    'horse_id',
    'sex_and_age',
    'burden_weight',
    'rider_id',
    'goal_time',
    'goal_time_dif',
    'time_value',
    'half_way_rank',
    'last_time',
    'odds',
    'popular',
    'horse_weight',
    'tame_time',
    'tamer_id',
    'owner_id'
]

def make_csv_from_html():
    for year in range(2008, now_datetime.year+1):
        make_csv_from_html_by_year(year)

def make_csv_from_html_by_year(year):
    save_race_csv = CSV_DIR+"/race-"+str(year)+".csv"
    horse_race_csv = CSV_DIR+"/horse-"+str(year)+".csv"
    if not ((os.path.isfile(save_race_csv)) and (os.path.isfile(horse_race_csv))): # まだcsvがなければ生成
        race_df = pd.DataFrame(columns=race_data_columns )
        horse_df = pd.DataFrame(columns=horse_data_columns )
        logger.info("saving csv (" + str(year) +")")
        total = 0;
        for month in range(1, 13):
            # race_html/year/month というディレクトリが存在すればappend, なければ何もしない
            html_dir = RACR_HTML_DIR+"/"+str(year)+"/"+str(month)
            if os.path.isdir(html_dir):
                file_list = os.listdir(html_dir) # get all file names
                total += len(file_list)
                logger.info(" appending " + str(len(file_list)) + " datas to csv (" + str(year)  +" "+ str(month)+ ")")
                for file_name in file_list:
                    with open(html_dir+"/"+file_name, "r") as f:
                        html = f.read()
                        list = file_name.split(".")
                        race_id = list[-2]
                        race_list, horse_list_list = get_rade_and_horse_data_by_html(race_id, html)
                        for horse_list in horse_list_list:
                            horse_se = pd.Series( horse_list, index=horse_df.columns)
                            horse_df = horse_df.append(horse_se, ignore_index=True)
                        race_se = pd.Series(race_list, index=race_df.columns )
                        race_df = race_df.append(race_se, ignore_index=True )

        race_df.to_csv(save_race_csv, header=True, index=False)
        horse_df.to_csv(horse_race_csv, header=True, index=False)
        logger.info(' (rows, columns) of race_df:\t'+ str(race_df.shape))
        logger.info(' (rows, columns) of horse_df:\t'+ str(horse_df.shape))
        logger.info("saved " + str(total) + " htmls to csv (" + str(year) +")")
    else:
        logger.info("already have csv (" + str(year) +")")

def get_rade_and_horse_data_by_html(race_id, html):
    race_list = [race_id]
    horse_list_list = []
    soup = BeautifulSoup(html, 'html.parser')

    # race基本情報
    data_intro = soup.find("div", class_="data_intro")
    race_list.append(data_intro.find("dt").get_text().strip("\n")) # race_round
    race_list.append(data_intro.find("h1").get_text().strip("\n")) # race_title
    race_details1 = data_intro.find("p").get_text().strip("\n").split("\xa0/\xa0")
    race_list.append(race_details1[0]) # race_course
    race_list.append(race_details1[1]) # weather
    race_list.append(race_details1[2]) # ground_status
    race_list.append(race_details1[3]) # time
    race_details2 = data_intro.find("p", class_="smalltxt").get_text().strip("\n").split(" ")
    race_list.append(race_details2[0]) # date
    race_list.append(race_details2[1]) # where_racecourse


    result_rows = soup.find("table", class_="race_table_01 nk_tb_common").findAll('tr') # レース結果
    # 上位3着の情報
    race_list.append(len(result_rows)-1) # total_horse_number
    for i in range(1,4):
        row = result_rows[i].findAll('td')
        race_list.append(row[1].get_text()) # frame_number_first or second or third
        race_list.append(row[2].get_text()) # horse_number_first or second or third


    # 払い戻し(単勝・複勝・三連複・3連単)
    pay_back_tables = soup.findAll("table", class_="pay_table_01")

    pay_back1 = pay_back_tables[0].findAll('tr') # 払い戻し1(単勝・複勝)
    race_list.append(pay_back1[1].find("td", class_="txt_r").get_text()) #tansyo
    hukuren = pay_back1[1].find("td", class_="txt_r")
    tmp = []
    for string in hukuren.strings:
        tmp.append(string)
    for i in range(3):
        try:
            race_list.append(tmp[i]) # hukuren_first or second or third
        except IndexError:
            race_list.append("0")

    pay_back2 = pay_back_tables[1].findAll('tr') # 払い戻し2(三連複・3連単)
    race_list.append(pay_back2[2].find("td", class_="txt_r").get_text()) #renhuku3
    try:
        race_list.append(pay_back2[3].find("td", class_="txt_r").get_text()) #rentan3
    except IndexError:
        race_list.append("0")

    # horse data
    for rank in range(1, len(result_rows)):
        horse_list = [race_id]
        result_row = result_rows[rank].findAll("td")
        # rank
        horse_list.append(result_row[0].get_text())
        # frame_number
        horse_list.append(result_row[1].get_text())
        # horse_number
        horse_list.append(result_row[2].get_text())
        # horse_id
        horse_list.append(result_row[3].find('a').get('href').split("/")[-2])
        # sex_and_age
        horse_list.append(result_row[4].get_text())
        # burden_weight
        horse_list.append(result_row[5].get_text())
        # rider_id
        horse_list.append(result_row[6].find('a').get('href').split("/")[-2])
        # goal_time
        horse_list.append(result_row[7].get_text())
        # goal_time_dif
        horse_list.append(result_row[8].get_text())
        # time_value(premium)
        horse_list.append(result_row[9].get_text())
        # half_way_rank
        horse_list.append(result_row[10].get_text())
        # last_time(上り)
        horse_list.append(result_row[11].get_text())
        # odds
        horse_list.append(result_row[12].get_text())
        # popular
        horse_list.append(result_row[13].get_text())
        # horse_weight
        horse_list.append(result_row[14].get_text())
        # tame_time(premium)
        horse_list.append(result_row[15].get_text())
        # 16:コメント、17:備考
        # tamer_id
        horse_list.append(result_row[18].find('a').get('href').split("/")[-2])
        # owner_id
        horse_list.append(result_row[19].find('a').get('href').split("/")[-2])

        horse_list_list.append(horse_list)

    return race_list, horse_list_list



#def update_csv():


if __name__ == '__main__':
    formatter = "%(asctime)s [%(levelname)s]\t%(message)s" # フォーマットを定義
    #formatter_func = "%(asctime)s\t[%(levelname)8s]\t%(message)s from %(func)" # フォーマットを定義
    logging.basicConfig(filename='logfile/'+OWN_FILE_NAME+'.logger.log', level=logging.INFO, format=formatter)
    
    logger.info("start making csv!")
    make_csv_from_html()
