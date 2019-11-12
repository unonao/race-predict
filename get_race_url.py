# raceの全情報のURLを入手する
"""
https://db.netkeiba.com/?pid=race_search_detail
検索窓から全レースを一覧表示
→レースへのリンクを取得
"""
import datetime
import pytz
now_datetime = datetime.datetime.now(pytz.timezone('Asia/Tokyo'))

import re
import time

import os
from os import path
OWN_FILE_NAME = path.splitext(path.basename(__file__))[0]
RACR_URL_DIR = "race_url"

import logging
formatter = "%(asctime)s [%(levelname)s]\t%(message)s" # フォーマットを定義
#formatter_func = "%(asctime)s\t[%(levelname)8s]\t%(message)s from %(func)" # フォーマットを定義
logging.basicConfig(filename='logfile/'+OWN_FILE_NAME+'.logger.log', level=logging.INFO, format=formatter)
logger = logging.getLogger(__name__) #ファイルの名前を渡す



from selenium import webdriver
from selenium.webdriver.support.ui import Select,WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
URL = "https://db.netkeiba.com/?pid=race_search_detail"
WAIT_SECOND = 5


def get_race_url():
    options = Options()
    options.add_argument('--headless')
    driver = webdriver.Chrome(chrome_options=options) # mac はbrewでインストールしたのでpathはok
    driver.implicitly_wait(10)
    # 去年までのデータ
    for year in range(2008, now_datetime.year):
        for month in range(1, 13):
            race_url_file = RACR_URL_DIR + "/" + str(year) + "-" + str(month) + ".txt" #保存先ファイル
            if not os.path.isfile(race_url_file): # まだ取得していなければ取得
                logger.info("getting urls ("+str(year) +" "+ str(month) + ")")
                get_race_url_by_year_and_mon(driver,year,month)
    # 先月までのデータ
    for year in range(now_datetime.year, now_datetime.year+1):
        for month in range(1, now_datetime.month):
            race_url_file = RACR_URL_DIR + "/" + str(year) + "-" + str(month) + ".txt" #保存先ファイル
            if not os.path.isfile(race_url_file): # まだ取得していなければ取得
                logger.info("getting urls ("+str(year) +" "+ str(month) + ")")
                get_race_url_by_year_and_mon(driver,year,month)
    # 今月分は毎回取得
    logger.info("getting urls ("+str(now_datetime.year) +" "+ str(now_datetime.month) + ")")
    get_race_url_by_year_and_mon(driver, now_datetime.year, now_datetime.month)

    driver.close()
    driver.quit()

def get_race_url_by_year_and_mon(driver, year, month):
    race_url_file = RACR_URL_DIR + "/" + str(year) + "-" + str(month) + ".txt" #保存先ファイル

    # URLにアクセス
    wait = WebDriverWait(driver,10)
    driver.get(URL)
    time.sleep(1)
    wait.until(EC.presence_of_all_elements_located)

    # 期間を選択
    start_year_element = driver.find_element_by_name('start_year')
    start_year_select = Select(start_year_element)
    start_year_select.select_by_value(str(year))
    start_mon_element = driver.find_element_by_name('start_mon')
    start_mon_select = Select(start_mon_element)
    start_mon_select.select_by_value(str(month))
    end_year_element = driver.find_element_by_name('end_year')
    end_year_select = Select(end_year_element)
    end_year_select.select_by_value(str(year))
    end_mon_element = driver.find_element_by_name('end_mon')
    end_mon_select = Select(end_mon_element)
    end_mon_select.select_by_value(str(month))

    # 競馬場をチェック
    for i in range(1,11):
        terms = driver.find_element_by_id("check_Jyo_"+ str(i).zfill(2))
        terms.click()

    # 表示件数を選択(20,50,100の中から最大の100へ)
    list_element = driver.find_element_by_name('list')
    list_select = Select(list_element)
    list_select.select_by_value("100")

    # フォームを送信
    frm = driver.find_element_by_css_selector("#db_search_detail_form > form")
    frm.submit()
    time.sleep(1)
    wait.until(EC.presence_of_all_elements_located)

    total_num_and_now_num = driver.find_element_by_xpath("//*[@id='contents_liquid']/div[1]/div[2]").text
    total_num = int(re.search(r'(.*)件中', total_num_and_now_num).group().strip("件中"))

    pre_url_num = 0
    if os.path.isfile(race_url_file):
        with open(race_url_file, mode='r') as f:
            pre_url_num = len(f.readlines())

    if total_num!=pre_url_num:
        with open(race_url_file, mode='w') as f:
            #tableからraceのURLを取得(ループしてページ送り)
            total = 0
            while True:
                time.sleep(1)
                wait.until(EC.presence_of_all_elements_located)

                all_rows = driver.find_element_by_class_name('race_table_01').find_elements_by_tag_name("tr")
                total += len(all_rows)-1
                for row in range(1, len(all_rows)):
                    race_href = all_rows[row].find_elements_by_tag_name("td")[4].find_element_by_tag_name("a").get_attribute("href")
                    f.write(race_href+"\n")

                # If the driver can click, do so. If not, break loop.
                try:
                    target = driver.find_elements_by_link_text("次")[0]
                    driver.execute_script("arguments[0].click();", target) #javascriptでクリック処理
                except IndexError:
                    break
        logging.info("got "+ str(total) +" urls of " + str(total_num) +" ("+str(year) +" "+ str(month) + ")")
    else:
        logging.info("already have " + str(pre_url_num) +" urls ("+str(year) +" "+ str(month) + ")")


if __name__ == '__main__':
    logger.info("start get race url!")
    get_race_url()
