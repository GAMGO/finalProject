# run_server.sh (ai 루트 폴더에 위치)
#!/bin/bash
set -e

# 1. API 서버 실행 전: database.py의 cursorclass 주석 해제 (복원)
# API 서버가 DB에서 데이터를 읽고 객체로 변환하는 데 필요한 설정이므로 필수입니다.
echo "🛠️ API 서버 실행을 위해 database.py 설정 복원 (주석 해제)..."
python modify_db_config.py uncomment

# 2. 최종 API 서버 실행
echo "🚀 Uvicorn API 서버 실행 중..."
# 0.0.0.0으로 바인딩하여 외부 접근이 가능하도록 합니다.
uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}