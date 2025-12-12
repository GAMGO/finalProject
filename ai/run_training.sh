# run_training.sh (ai 루트 폴더에 위치)

#!/bin/bash
set -e # 명령어 오류 발생 시 즉시 종료

echo "📦 필요한 라이브러리 설치 중..."
pip install -r requirements.txt

# 1. 더미 데이터 추가 (실제 스크립트 이름으로 변경)
echo "🧱 더미 데이터 추가 스크립트 실행 중..."
# python dummy_data_script.py # 해당 스크립트가 있다면 주석 해제하여 사용

# 2. 학습 전: database.py의 cursorclass 주석 처리
echo "🛠️ 학습을 위해 database.py 설정 수정 (주석 처리)..."
python modify_db_config.py comment

# 3. 학습 스크립트 실행
echo "🧠 리뷰 감성 분석 및 배치 작업 실행 중..."
python sentiment_batch_reviews.py

echo "🧠 상점 추천 모델 학습 실행 중..."
python train_stall_recommender.py

# 4. 학습 후: database.py의 cursorclass 주석 해제
echo "🛠️ API 서버 실행을 위해 database.py 설정 복원 (주석 해제)..."
python modify_db_config.py uncomment

# 5. 최종 API 서버 실행
echo "🚀 Uvicorn API 서버 실행 중..."
# --reload 옵션은 배포 환경에서는 권장되지 않으므로 제거합니다.
# 0.0.0.0으로 바인딩하여 외부 접근이 가능하도록 합니다.
uvicorn main:app --host 0.0.0.0 --port 8000