from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.callbacks.base import BaseCallbackHandler
import streamlit as st
import os
from dotenv import load_dotenv
import sqlite3
from langchain_google_genai import GoogleGenerativeAI
import sqlite3
import json
from sentence_transformers import SentenceTransformer, util
import sqlite3
import numpy as np
from langchain_core.prompts import PromptTemplate
import ast
import random
import re
from geopy import distance


load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")


model = SentenceTransformer('intfloat/multilingual-e5-large-instruct')


# get prompts
prompt_template_branch_splitter_path = "./prompt/prompt1-question-classification.txt"
with open(prompt_template_branch_splitter_path, 'r', encoding='utf-8') as file:
    prompt_template_branch_splitter_text = file.read()

prompt_template_card_path = "./prompt/prompt2-1-card.txt"
with open(prompt_template_card_path, 'r', encoding='utf-8') as file:
    prompt_template_card_text = file.read()

prompt_template_normal_path = "./prompt/prompt2-2-recommendation.txt"
with open(prompt_template_normal_path, 'r', encoding='utf-8') as file:
    prompt_template_normal_text = file.read()

prompt_template_tour_path = "./prompt/prompt3-3-tour.txt"
with open(prompt_template_tour_path, 'r', encoding='utf-8') as file:
    prompt_template_tour_text = file.read()


# gemini setting
llm = GoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=api_key,
    temperature=0.1,
    streaming=True)


#######################################################################################################################################


def send_message(message, role):

    with st.chat_message(role):
        st.markdown(message)


st.title("🍊 맛르방")

send_message("""
             혼저옵서예~\n
             **저에게 무엇이든 물어보세요!** \n
             '맛집 알려줘! 도민들이 자주 가고, 주차하기 편한 곳으로!', '20대 남자 넷이서 갈만한 횟집 있어?' 등 뭐든 편하게 물어보시면 정확하게 답해드릴게요!
             """, "ai")


#######################################################################################################################################


# branch splitter prompt template
prompt_template_branch_splitter = ChatPromptTemplate.from_messages(
    [
        (
            "system", "{context}.",
        ),
        ("human", "{question}"),
    ]
)

# card prompt template
prompt_template_card = ChatPromptTemplate.from_messages(
    [
        (
            "system", "{context}.",
        ),
        ("human", "{question}"),
    ]
)

# normal prompt template
prompt_template_normal = ChatPromptTemplate.from_messages(
    [
        (
            "system", "{context}.",
        ),
        ("human", "{question}"),
    ]
)


# tour prompt template
prompt_template_tour = ChatPromptTemplate.from_messages(
    [
        (
            "system", "{context}",
        ),
        ("human", "{question}"),
    ]
)

# final explain prompt template
final_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system", f"""내가 제시한 음식점들을 각각 두 줄로 설명해.

            AMENITY와 BOSS_TIP, keywords는 최대한 포함시켜야 해. 다만, 자연스럽게 설명해야 해. 'AMENITY', 'keywords'를 직접적으로 명시해서는 안돼. 만약 '정보없음'과 같이 제대로 된 데이터가 아니라면 해당 부분은 넘어가야 해.

            Review의 경우에는 아래와 같이 말해야 해

            '''
            제가 리뷰들을 요약해봤어요!
            '제공한 리뷰 요약'
            '''

            만약 아무런 값도 들어오지 않았다면, '해당 조건을 가진 음식점을 찾지 못했어요. 다시 검색해주세요.'라고 답해야 해.
            """,
        ),
        ("human", "{question}"),
    ]
)

# chains
# 각각 체인들 여따 정의

chain_branch_splitter = (
    {
        "question": RunnablePassthrough(), "context": RunnablePassthrough()
    }
    | prompt_template_branch_splitter
    | llm)
chain_card = (
    {
        "question": RunnablePassthrough(), "context": RunnablePassthrough()
    }
    | prompt_template_card
    | llm)
chain_normal = (
    {
        "question": RunnablePassthrough(), "context": RunnablePassthrough()
    }
    | prompt_template_normal
    | llm)
chain_tour = (
    {
        "question": RunnablePassthrough(), "context": RunnablePassthrough()
    }
    | prompt_template_tour
    | llm)

chain_final = (
    {
        "question": RunnablePassthrough()
    }
    | final_prompt
    | llm)

#####################################################################################################################################################


def convert_string_to_dict(input_string):
    """
    문자열을 딕셔너리로 변환하고 'Null' 값을 None으로 변경하는 함수

    Args:
        input_string (str): 변환할 문자열

    Returns:
        dict: 변환된 딕셔너리
    """
    # 작은따옴표를 큰따옴표로 변경하여 JSON 형식으로 만듦
    json_string = input_string.replace("'", '"')

    try:
        # JSON 파싱 시도
        result = json.loads(json_string)
    except json.JSONDecodeError:
        # JSON 파싱 실패시 ast.literal_eval 사용
        result = ast.literal_eval(input_string)

    # 'Null' 값을 None으로 변경
    for key, value in result.items():
        if value == 'Null':
            result[key] = None

    return result


def recommendation(llm_output: dict):
    conn = sqlite3.connect('./final.db')
    cursor = conn.cursor()

    # 기본 쿼리
    query = "SELECT id, keywords_embeddings FROM Information WHERE Closed = 0"
    conditions = []

    if llm_output["name"] is not None:
        conditions.append(f"MCT_NAVER_NAME LIKE '%{llm_output['name']}%'")

    if llm_output["popular"] is not None:
        conditions.append("UE_CNT_GRP IN ('4_50~75%','5_75~90%', '6_90% 초과')")

    if llm_output["is_cheap"] is not None:
        conditions.append("UE_AMT_PER_TRSN_GRP IN ('2_10~25%', '1_상위 10% 이하')")

    if llm_output["is_expensive"] is not None:
        conditions.append(
            "UE_AMT_PER_TRSN_GRP IN ('5_75~90%', '6_90% 초과')")

    # 만약 is local이 none이 아니라면 0.5 이상만 뽑아내기 LOCAL_UE_CNT_RAT 이용 float 변수
    if llm_output["is_local"] is not None:
        conditions.append("LOCAL_UE_CNT_RAT >= 0.69")

    # 만약 is popular for female이 none이 아니라면 0.x 이상만 뽑아내기, 실제로 들가있는 값들 보면서 진행..RC_M12_FME_CUS_CNT_RAT
    if llm_output["is_popular_for_male"] is not None:
        conditions.append("RC_M12_MAL_CUS_CNT_RAT >= 0.621")

    if llm_output["is_popular_for_female"] is not None:
        conditions.append("RC_M12_FME_CUS_CNT_RAT >= 0.466")

    if llm_output["is_popular_for_20"] is not None:
        conditions.append("RC_M12_AGE_UND_20_CUS_CNT_RAT >= 0.16")

    if llm_output["is_popular_for_30"] is not None:
        conditions.append("RC_M12_AGE_30_CUS_CNT_RAT >= 0.27")

    if llm_output["is_popular_for_40"] is not None:
        conditions.append("RC_M12_AGE_40_CUS_CNT_RAT >= 0.306")

    if llm_output["is_popular_for_50"] is not None:
        conditions.append("RC_M12_AGE_50_CUS_CNT_RAT >= 0.256")

    if llm_output["is_popular_for_over_60"] is not None:
        conditions.append("RC_M12_AGE_OVR_60_CUS_CNT_RAT >= 0.12")

    if llm_output["work_day"] is not None:
        if llm_output["work_day"] == "주말":
            conditions.append(
                "((WT LIKE '%토%' AND WT LIKE '%일%') OR WT LIKE '%매일%')")
        else:
            conditions.append(f"WT LIKE '%{llm_output['work_day']}%'")

    if llm_output["work_time"] is not None:
        if llm_output["work_day"] == "밤":
            conditions.append(
                "(HR_18_22_UE_CNT_RAT >= 0.15) AND (HR_23_4_UE_CNT_RAT >= 0.1) ")
        elif llm_output["work_day"] == "아침":
            conditions.append("HR_5_11_UE_CNT_RAT >= 0.1")
        elif llm_output["work_day"] == "저녁":
            conditions.append("HR_18_22_UE_CNT_RAT >= 0.2")
        elif llm_output["work_day"] == "새벽":
            conditions.append(
                "(HR_5_11_UE_CNT_RAT >= 0.05) AND (HR_23_4_UE_CNT_RAT >= 0.1)")

    if llm_output["closed_day"] is not None:
        if llm_output["closed_day"] == "주말":
            conditions.append(
                "((CLSD LIKE '%토%' AND CLSD LIKE '%일%') OR CLSD LIKE '%매일%')")
        else:
            conditions.append(f"CLSD LIKE '%{llm_output['closed_day']}%'")

    if llm_output["wheelchair_access"] is not None:
        conditions.append("wheelchair_access  = 1")

    if llm_output["restaurant_type"] is not None:
        conditions.append(f"MCT_TYPE LIKE '%{llm_output['restaurant_type']}%'")
    # -> MCT_NAVER_TYPE이 없으면 MCT_TYPE으로 변경해야함
    # RC_M12_MAL_CUS_CNT_RAT
    if conditions:
        query += " AND " + " AND ".join(conditions)
    # 쿼리 실행
    cursor.execute(query)
    results = cursor.fetchall()

    # 연결 종료
    conn.close()

    return results


def simm_score(recommendation_no_scoring_result, return_num=10):
    data = []
    data = [(id_, np.frombuffer(value, dtype=np.float32))
            for id_, value in recommendation_no_scoring_result if value is not None]

    user_embedding = model.encode(message)

    cosine_scores = []
    for id_, embeddings in data:
        score = util.pytorch_cos_sim(user_embedding, embeddings)
        cosine_scores.append((id_, score.item()))

    cosine_scores.sort(key=lambda x: x[1], reverse=True)

    top_results = []
    # 상위 {return_num}만큼 반환,
    for id_,  score in cosine_scores[:return_num]:
        top_results.append(
            {'id': id_, 'simm': score})

    return top_results


def get_restaurant_info(top_):
    # 데이터베이스 연결
    conn = sqlite3.connect('./final.db')
    cursor = conn.cursor()

    # 결과를 저장할 리스트
    results = []

    # 각 id에 대해 정보 조회
    for item in top_:
        restaurant_id = item['id']

        # SQL 쿼리 작성
        query = f"""
        SELECT MCT_NM, MCT_TYPE, ADDR, WT, PHONE, AMENITY, placeID, PAYMENT, BOSS_TIP,
               keywords, RC_M12_MAL_CUS_CNT_RAT, RC_M12_FME_CUS_CNT_RAT,
               RC_M12_AGE_UND_20_CUS_CNT_RAT, RC_M12_AGE_30_CUS_CNT_RAT,
               RC_M12_AGE_40_CUS_CNT_RAT, RC_M12_AGE_50_CUS_CNT_RAT,
               RC_M12_AGE_OVR_60_CUS_CNT_RAT
        FROM Information
        WHERE id = ?
        """

        # 쿼리 실행
        cursor.execute(query, (restaurant_id,))

        # 결과 가져오기
        result = cursor.fetchone()
        if result:
            # 컬럼 이름 가져오기
            column_names = [description[0]
                            for description in cursor.description]
            # 결과를 딕셔너리 형태로 변환
            restaurant_info = dict(zip(column_names, result))
            results.append(restaurant_info)

    # 연결 종료
    conn.close()

    return results


def extract_sql_query(text):
    # Regular expression to match SQL query starting with SELECT and ending with ;
    pattern = r"(SELECT.*?;)"
    # Match across multiple lines
    matches = re.findall(pattern, text, re.DOTALL)
    return matches


def get_card_restaurant_info(sql_query):
    conn = sqlite3.connect('./final.db')
    cursor = conn.cursor()

    cursor.execute(sql_query)
    result = cursor.fetchone()

    return result


def get_near_tourpoint(search_area):
    # SQLite 데이터베이스 연결
    db_path = './final.db'  # 데이터베이스 파일 경로
    connection = sqlite3.connect(db_path)

    try:
        cursor = connection.cursor()

        # 1. 관광지 좌표 가져오기
        query = "SELECT Latitude, Longitude FROM Tour WHERE AREA_NM LIKE ?;"
        cursor.execute(query, (f"%{search_area}%",))  # 부분 일치 검색
        rows = cursor.fetchall()

        if not rows:
            raise ValueError(f"'{search_area}'에 해당하는 관광지를 찾을 수 없습니다.")

        # 기준 좌표 설정 (첫 번째 결과 사용)
        reference_point = (rows[0][0], rows[0][1])  # 위도, 경도

        # 2. 맛집 정보 가져오기
        query = "SELECT id, Latitude, Longitude FROM Information WHERE Longitude != 0;"
        cursor.execute(query)
        rows = cursor.fetchall()

        # 3. 거리 계산 및 정렬
        results = []
        for row in rows:
            id, latitude, longitude = row
            location = (latitude, longitude)  # 위도, 경도
            dist = distance.distance(reference_point, location).km
            results.append((id, location, dist))

        # 거리 기준 정렬
        results.sort(key=lambda x: x[2])  # 거리(dist)를 기준으로 오름차순 정렬

        # 상위 10개 결과 추출
        top_10 = [{'id': result[0]} for result in results[:10]]

        return top_10

    except Exception as e:
        print("Error:", e)
        return []
    finally:
        connection.close()


message = st.chat_input("질문을 입력해주세요!")

if message:
    send_message(message, "human")

    chain_branch_splitter_result = chain_branch_splitter.invoke(
        {"context": prompt_template_branch_splitter_text, "question": message})

    print("chain_branch_splitter output:", chain_branch_splitter_result)

    if "신한카드" in chain_branch_splitter_result:
        chain_card_result_sql = chain_branch_splitter.invoke(
            {"context": prompt_template_card_text, "question": message})
        print(type(chain_card_result_sql))

        if '```sql' in chain_card_result_sql:
            start = chain_card_result_sql.find('sql') + len('sql\n')
            end = chain_card_result_sql.find('```', start)
            chain_card_result_sql = chain_card_result_sql[start:end].strip()
            print("json to dict parsing!")
            print(chain_card_result_sql)
        else:
            print(chain_card_result_sql)

        chain_card_sql_query = extract_sql_query(chain_card_result_sql)

        one_restaurant_get_card_restaurant = get_card_restaurant_info(
            chain_card_result_sql)

        only_one_for_card = [{"id": one_restaurant_get_card_restaurant[0]}]

        restaurant_info_for_card = get_restaurant_info(only_one_for_card)

        if restaurant_info_for_card:
            for data_dict in restaurant_info_for_card:
                if data_dict['MCT_NM'] is not None:
                    st.sidebar.markdown(
                        f"### [{data_dict['MCT_NM']}]("
                        f"https://map.naver.com/p/search/{data_dict['MCT_NM'].replace(
                            ' ', '')}/place/{data_dict['placeID']}?c=15.00,0,0,0,dh&isCorrectAnswer=true)"
                    )

                if data_dict['ADDR'] is not None:
                    st.sidebar.write(f"주소: {data_dict['ADDR']}")
                if data_dict['PHONE'] is not None:
                    st.sidebar.write(f"전화번호: {data_dict['PHONE']}")
                if data_dict['AMENITY'] is not None:
                    amenities = data_dict['AMENITY'].replace("|", " | ")
                    st.sidebar.write(f"편의시설: {amenities}")
                if data_dict['PAYMENT'] is not None:
                    st.sidebar.write(f"추가 결제 수단: {data_dict['PAYMENT']}")
                if data_dict['BOSS_TIP'] is not None:
                    with st.sidebar.expander("사장님 팁 보기"):
                        st.sidebar.write(data_dict['BOSS_TIP'])
                if data_dict['MCT_NM'] is not None:
                    st.sidebar.write("---")

        final_string = ""
        data_dict = restaurant_info_for_card[0]  # 요소가 하나이므로 직접 접근

        name = data_dict['MCT_NM'] if data_dict['MCT_NM'] is not None else "정보 없음"
        mct_type = data_dict['MCT_TYPE'] if data_dict['MCT_TYPE'] is not None else "정보 없음"
        addr = data_dict['ADDR'] if data_dict['ADDR'] is not None else "정보 없음"
        wt = data_dict['WT'] if data_dict['WT'] is not None else "정보 없음"
        amenity = data_dict['AMENITY'] if data_dict['AMENITY'] is not None else "정보 없음"
        payment = data_dict['PAYMENT'] if data_dict['PAYMENT'] is not None else "정보 없음"
        keywords = data_dict['keywords'] if data_dict['keywords'] is not None else "정보 없음"

        final_string += f"""1. 이름: '{name}', 업종: '{mct_type}', 주소: '{addr}', 영업시간: '{
            wt}', 시설정보: '{amenity}', 추가지불수단: '{payment}', 리뷰 요약: '{keywords}'\n\n"""

        chain_final_result = chain_final.invoke(
            {"question": final_string})

        send_message(chain_final_result, "ai")

    if "맛집추천" in chain_branch_splitter_result:
        chain_normal_result_recommend_func_parameter = chain_branch_splitter.invoke(
            {"context": prompt_template_normal_text, "question": message})

        print(chain_normal_result_recommend_func_parameter)
        response = chain_normal_result_recommend_func_parameter
        if '```json' in response:
            start = response.find('json') + len('json\n')
            end = response.find('```', start)
            response = response[start:end].strip()
            print("json to dict parsing!")
            print(response)
        else:
            print(response)
        converted_dict = convert_string_to_dict(response)

        output = recommendation(converted_dict)

        top_ = simm_score(output)
        print(top_)
        print(type(top_))

        restaurant_info = get_restaurant_info(top_)
        print("===="*5)
        print()
        print(restaurant_info)
        print(type(restaurant_info))
        print("")
        print("")
        print("")
        if restaurant_info:
            for data_dict in restaurant_info:
                if data_dict['MCT_NM'] is not None:
                    st.sidebar.markdown(
                        f"### [{data_dict['MCT_NM']}]("
                        f"https://map.naver.com/p/search/{data_dict['MCT_NM'].replace(
                            ' ', '')}/place/{data_dict['placeID']}?c=15.00,0,0,0,dh&isCorrectAnswer=true)"
                    )

                if data_dict['ADDR'] is not None:
                    st.sidebar.write(f"주소: {data_dict['ADDR']}")
                if data_dict['PHONE'] is not None:
                    st.sidebar.write(f"전화번호: {data_dict['PHONE']}")
                if data_dict['AMENITY'] is not None:
                    amenities = data_dict['AMENITY'].replace("|", " | ")
                    st.sidebar.write(f"편의시설: {amenities}")
                if data_dict['PAYMENT'] is not None:
                    st.sidebar.write(f"추가 결제 수단: {data_dict['PAYMENT']}")
                if data_dict['BOSS_TIP'] is not None:
                    with st.sidebar.expander("사장님 팁 보기"):
                        st.sidebar.write(data_dict['BOSS_TIP'])
                if data_dict['MCT_NM'] is not None:
                    st.sidebar.write("---")

        final_string = ""
        # enumerate를 사용하여 인덱스를 추가
        for idx, data_dict in enumerate(restaurant_info, start=1):
            if data_dict['MCT_NM'] is not None:
                name = data_dict['MCT_NM']
            else:
                name = "정보 없음"

            if data_dict['MCT_TYPE'] is not None:
                mct_type = data_dict['MCT_TYPE']
            else:
                mct_type = "정보 없음"

            if data_dict['ADDR'] is not None:
                addr = data_dict['ADDR']
            else:
                addr = "정보 없음"

            if data_dict['WT'] is not None:
                wt = data_dict['WT']
            else:
                wt = "정보 없음"

            if data_dict['AMENITY'] is not None:
                amenity = data_dict['AMENITY']
            else:
                amenity = "정보 없음"

            if data_dict['PAYMENT'] is not None:
                payment = data_dict['PAYMENT']
            else:
                payment = "정보 없음"

            if data_dict['keywords'] is not None:
                keywords = data_dict['keywords']
            else:
                keywords = "정보 없음"

            final_string += f"""{idx}. 이름: '{name}', 업종: '{mct_type}', 주소: '{addr}', 영업시간: '{
                wt}', 시설정보: '{amenity}', 추가지불수단: '{payment}', 리뷰 요약: '{keywords}'\n\n"""

        chain_final_result = chain_final.invoke(
            {"question": final_string})

        send_message(chain_final_result, "ai")

    if "관광지맛집" in chain_branch_splitter_result:
        chain_tour_nearby_restaurant = chain_branch_splitter.invoke(
            {"context": prompt_template_tour_text, "question": message})
        print(type(chain_tour_nearby_restaurant))
        print(chain_tour_nearby_restaurant)

        top_restaurants_tour = get_near_tourpoint("만장굴")
        print(top_restaurants_tour)

        restaurant_info_tour = get_restaurant_info(top_restaurants_tour)
        print("===="*5)
        print()
        print(restaurant_info_tour)
        print(type(restaurant_info_tour))
        print("")
        print("")
        print("")
        if restaurant_info_tour:
            for data_dict in restaurant_info_tour:
                if data_dict['MCT_NM'] is not None:
                    st.sidebar.markdown(
                        f"### [{data_dict['MCT_NM']}]("
                        f"https://map.naver.com/p/search/{data_dict['MCT_NM'].replace(
                            ' ', '')}/place/{data_dict['placeID']}?c=15.00,0,0,0,dh&isCorrectAnswer=true)"
                    )

                if data_dict['ADDR'] is not None:
                    st.sidebar.write(f"주소: {data_dict['ADDR']}")
                if data_dict['PHONE'] is not None:
                    st.sidebar.write(f"전화번호: {data_dict['PHONE']}")
                if data_dict['AMENITY'] is not None:
                    amenities = data_dict['AMENITY'].replace("|", " | ")
                    st.sidebar.write(f"편의시설: {amenities}")
                if data_dict['PAYMENT'] is not None:
                    st.sidebar.write(f"추가 결제 수단: {data_dict['PAYMENT']}")
                if data_dict['BOSS_TIP'] is not None:
                    with st.sidebar.expander("사장님 팁 보기"):
                        st.sidebar.write(data_dict['BOSS_TIP'])
                if data_dict['MCT_NM'] is not None:
                    st.sidebar.write("---")

        final_string = ""
        # enumerate를 사용하여 인덱스를 추가
        for idx, data_dict in enumerate(restaurant_info_tour, start=1):
            if data_dict['MCT_NM'] is not None:
                name = data_dict['MCT_NM']
            else:
                name = "정보 없음"

            if data_dict['MCT_TYPE'] is not None:
                mct_type = data_dict['MCT_TYPE']
            else:
                mct_type = "정보 없음"

            if data_dict['ADDR'] is not None:
                addr = data_dict['ADDR']
            else:
                addr = "정보 없음"

            if data_dict['WT'] is not None:
                wt = data_dict['WT']
            else:
                wt = "정보 없음"

            if data_dict['AMENITY'] is not None:
                amenity = data_dict['AMENITY']
            else:
                amenity = "정보 없음"

            if data_dict['PAYMENT'] is not None:
                payment = data_dict['PAYMENT']
            else:
                payment = "정보 없음"

            if data_dict['keywords'] is not None:
                keywords = data_dict['keywords']
            else:
                keywords = "정보 없음"

            final_string += f"""{idx}. 이름: '{name}', 업종: '{mct_type}', 주소: '{addr}', 영업시간: '{
                wt}', 시설정보: '{amenity}', 추가지불수단: '{payment}', 리뷰 요약: '{keywords}'\n\n"""

        chain_final_result = chain_final.invoke(
            {"question": final_string})

        send_message(chain_final_result, "ai")
