import sqlite3
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import re
from selenium.webdriver.firefox.service import Service as FirefoxService
from webdriver_manager.firefox import GeckoDriverManager
from selenium.common.exceptions import TimeoutException, NoSuchElementException


def preprocess_mct_nm(mct_nm):
    if mct_nm.startswith('('):
        mct_nm = re.sub(r'^\(.*?\)', '', mct_nm).strip()

    if ' ' not in mct_nm:
        mct_nm += " 제주"

    return mct_nm


def get_review_data(driver):
    try:
        total_review_text = WebDriverWait(driver, 10).until(
            EC.visibility_of_element_located(
                (By.XPATH, '/html/body/div[2]/div[2]/div[2]/div[1]/div[1]/div[2]/div/div[2]/a[1]/span[2]'))
        ).text

        total_review_num = int(re.search(r'\d+', total_review_text).group())
    except Exception as e:
        print("총 리뷰 수를 가져오는 중 오류 발생:", e)
        total_review_num = 0

    print("total_review_num", total_review_num)

    try:
        star_mean = float(WebDriverWait(driver, 10).until(
            EC.visibility_of_element_located(
                (By.XPATH, '/html/body/div[2]/div[2]/div[2]/div[1]/div[1]/div[2]/div/div[2]/a[1]/span[1]'))
        ).text)
    except Exception as e:
        print("별점을 가져오는 중 오류 발생:", e)
        star_mean = 0.0

    print("star_mean", star_mean)

    try:
        WebDriverWait(driver, 2).until(
            EC.presence_of_element_located(
                (By.XPATH, '/html/body/div[2]/div[2]/div[2]/div[8]/div[2]'))
        )
    except Exception as e:
        print("리뷰 섹션을 기다리는 중 오류 발생:", e)

    good_taste = good_price = good_parking = good_facilities = good_kindness = good_mood = 0

    try:
        div_element_review = driver.find_element(
            By.XPATH, '/html/body/div[2]/div[2]/div[2]/div[8]/div[2]')

        span_elements = div_element_review.find_elements(By.TAG_NAME, 'span')

        review_type = None
        score = 0

        for span in span_elements:
            if 'txt_likepoint' in span.get_attribute('class'):
                review_type = span.text
            elif 'num_likepoint' in span.get_attribute('class'):
                score = int(span.text)

            if review_type == "맛":
                good_taste += score
            elif review_type == "주차":
                good_parking += score
            elif review_type == "분위기":
                good_mood += score
            elif review_type == "친절":
                good_kindness += score
            elif review_type == "가성비":
                good_price += score
            elif review_type == "시설":
                good_facilities += score

    except Exception as e:
        print("리뷰 데이터 처리 중 오류 발생:", e)

    return total_review_num, star_mean, good_taste, good_price, good_parking, good_facilities, good_kindness, good_mood


def update_data_to_db(data):
    conn = sqlite3.connect('./kakaomap.db')
    cursor = conn.cursor()

    cursor.execute('''
        UPDATE kakao_map
        SET total_review_num = ?,
            star_mean = ?,
            good_taste = ?,
            good_price = ?,
            good_parking = ?,
            good_facilities = ?,
            good_kindness = ?,
            good_mood = ?,
            error_code = ?
        WHERE restaurant_id = ?
    ''', (data[3], data[4], data[5], data[6], data[7], data[8], data[9], data[10], 0, data[0]))

    conn.commit()
    conn.close()


def update_error_to_db(data):
    conn = sqlite3.connect('./kakaomap.db')
    cursor = conn.cursor()

    cursor.execute('''
        UPDATE kakao_map
        SET error_code = ?
        WHERE restaurant_id = ?
    ''', (data[1], data[0]))

    conn.commit()
    conn.close()


def fetch_mct_nm_and_addr_from_db():
    print("in fetch_mct_nm_and_addr_from_db")
    conn = sqlite3.connect('./kakaomap.db')
    print("create conn")
    cursor = conn.cursor()
    print("create cursor")

    cursor.execute(
        """
        SELECT MCT_NM, ADDR, restaurant_id
        FROM kakao_map
        where error_code is null
        """)
    results = cursor.fetchall()
    print("fetchall")
    print(results)
    conn.close()

    return [(mct_nm[0], mct_nm[1], mct_nm[2]) for mct_nm in results]


driver = webdriver.Firefox(service=FirefoxService("./gecko/geckodriver"))
try:
    driver.get('https://map.kakao.com/')

    mct_nms_n_addr = fetch_mct_nm_and_addr_from_db()

    for mct_nm, addr, restaurant_id in mct_nms_n_addr:
        try:
            mct_nm = preprocess_mct_nm(mct_nm)

            print("\n\n"+mct_nm, "검색중...")
            search_box = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located(
                    (By.XPATH, '//*[@id="search.keyword.query"]'))
            )
            search_box.clear()
            search_box.send_keys(mct_nm)

            search_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable(
                    (By.XPATH, '//*[@id="search.keyword.submit"]'))
            )
            driver.execute_script(
                "arguments[0].scrollIntoView();", search_button)
            driver.execute_script("arguments[0].click();", search_button)
            time.sleep(1)
            print("검색 완료... 총 검색 수 가져오는 중...")
            num_results = int(WebDriverWait(driver, 2).until(
                EC.presence_of_element_located(
                    (By.ID, "info.search.place.cnt"))
            ).text)

            print("총 검색 수:", num_results)

            if num_results > 0:
                place_items_xpath = "//li[contains(@class, 'PlaceItem')]"
                place_items = WebDriverWait(driver, 3).until(
                    EC.presence_of_all_elements_located(
                        (By.XPATH, place_items_xpath))
                )

                for n, place_item in enumerate(place_items):
                    try:
                        get_addr_xpath = ".//div[5]/div[2]/p[2]"
                        get_addr = place_item.find_element(
                            By.XPATH, get_addr_xpath).text

                        get_addr_bunji = re.sub(r'\(.*?\)\s*', '', get_addr)
                        print(f"{n + 1}/{len(place_items)} 처리 중...",
                              "|", get_addr_bunji, "| in | ", addr, "|")

                        if get_addr_bunji in addr:
                            review_link_xpath = ".//div[4]/a"
                            review_link = place_item.find_element(
                                By.XPATH, review_link_xpath).get_attribute('href')
                            driver.get(review_link)

                            print("moved to review page!")

                            total_review_num, star_mean, good_taste, good_price, good_parking, good_facilities, good_kindness, good_mood = get_review_data(
                                driver)

                            update_data_to_db((restaurant_id, mct_nm, addr, total_review_num, star_mean, good_taste,
                                               good_price, good_parking, good_facilities, good_kindness, good_mood, 0))

                            print(
                                f"총 리뷰 수: {total_review_num}, 평균 별점: {star_mean}, 맛: {good_taste}, 주차: {good_parking}, 친절: {good_kindness}, 가성비: {good_price}, 시설: {good_facilities}, 분위기: {good_mood}\n\n")

                            driver.back()
                            time.sleep(3)

                        else:
                            print("==== not exist ====")
                            update_error_to_db(
                                (restaurant_id, 105))  # not exist

                    except Exception as inner_e:
                        print(f"주소 가져오기 중 예외 발생: {inner_e}")
                        update_error_to_db((restaurant_id, 103))

            else:
                print("error_code 101")
                update_error_to_db((restaurant_id, 101))

        except Exception as e:
            print(f"식당 검색 중 예외 발생: {e}")
            update_error_to_db((restaurant_id, 102))

except Exception as e:
    print(f"errrr {e}")

finally:
    driver.quit()
