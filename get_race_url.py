# raceの全情報のURLを入手する
"""
https://db.netkeiba.com/?pid=race_search_detail
検索窓から全レースを一覧表示
→レースへのリンクを取得
"""

from os import path
OWN_FILE_NAME = path.splitext(path.basename(__file__))[0]

import logging
logger = logging.getLogger(__name__) # get_race_url.py の名前を渡す
formatter = '%(levelname)s : %(asctime)s : %(message)s' # フォーマットを定義
logging.basicConfig(filename='logfile/'+OWN_FILE_NAME+'.logger.log', level=logging.INFO, format=formatter)

from selenium import webdriver
from selenium.webdriver.support.ui import Select,WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
URL = "https://db.netkeiba.com/?pid=race_search_detail"
WAIT_SECOND = 5


def get_race_url():
    driver = webdriver.Chrome() # mac はbrewでインストール
    driver.implicitly_wait(10)
    for year in range(2008, 2020):
        for month in range(1, 13):
            try:
                get_race_url_by_year_and_mon(driver,year,month)
                logging.info("have gotten "+ str(year) +"/" + str(month) +" data")
            except:
                logging.warning("cannot get "+ str(year) +"/" + str(month) +" data")
    driver.close()
    driver.quit()

def get_race_url_by_year_and_mon(driver, year, month):
    # URLにアクセス
    driver.get(URL)

    WebDriverWait(driver, 15).until(EC.presence_of_all_elements_located)

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

    # 表示件数を選択(20,50,100の中から最大の100へ)
    list_element = driver.find_element_by_name('list')
    list_select = Select(list_element)
    list_select.select_by_value("100")

    # フォームを送信
    frm = driver.find_element_by_css_selector("#db_search_detail_form > form")
    frm.submit()

    WebDriverWait(driver, 15).until(EC.presence_of_all_elements_located)

    #リストからURLを取得(ページ送り含む)
    

if __name__ == '__main__':
    get_race_url()
