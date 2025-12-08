파이썬 3.11 설치

cd ai
#{username}에 컴퓨터 이름 복붙
"C:\Users\{username}\AppData\Local\Programs\Python\Python311\python.exe" -m venv venv

venv\Scripts\activate.bat

pip install -r requirements.txt

더미데이터 추가

학습 전 database.py의 cursorclass 주석처리

python sentiment_batch_reviews.py

python train_stall_recommender.py

학습 후 database.py의 cursorclass 주석풀기

uvicorn main:app --reload

api 호출방법

리뷰 요약 input
http://127.0.0.1:8000/api/stores/상점번호/summary

상점추천 input POST
http://127.0.0.1:8000/recommend/route
JSON 
{
    "start":{"lat":37.891537,"lng":127.061186},
    "waypoints":[{"lat":37.791537,"lng":127.161186}],
    "end":{"lat":37.527982,"lng":127.273066},
    "user_id":10
}