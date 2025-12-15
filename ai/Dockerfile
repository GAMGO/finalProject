# Dockerfile (ai 루트 폴더에 위치)

# 베이스 이미지: Python 3.11-slim 사용 (가장 가벼운 환경)
FROM python:3.11-slim

# 작업 디렉토리 설정
WORKDIR /app

# 1. 모든 프로젝트 파일 복사
# (main.py, requirements.txt, run_server.sh, modify_db_config.py 등 API 구동에 필요한 모든 파일)
COPY . /app

# 2. 필수 라이브러리 설치
# (uvicorn, fastapi, database access 라이브러리 등 포함)
RUN pip install -r requirements.txt

# 3. dos2unix 설치 및 run_server.sh 변환, 실행 권한 부여
# * dos2unix는 Linux/Unix 환경에서 Windows(CRLF) 개행 문자 문제를 방지하기 위해 권장됩니다.
RUN apt-get update && \
    apt-get install -y dos2unix && \
    dos2unix run_server.sh && \
    chmod +x run_server.sh && \
    rm -rf /var/lib/apt/lists/*

# 4. 포트 노출
EXPOSE 8000

# 5. CMD 명령: API 서버 실행 전용 스크립트 호출
# 컨테이너 시작 시 이 명령이 실행되어 Uvicorn 서버를 띄웁니다.
CMD ["/bin/bash", "./run_server.sh"]