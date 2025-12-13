# Dockerfile (ai 루트 폴더에 위치)

# 베이스 이미지: Python 3.11 사용
FROM python:3.11-slim

# 작업 디렉토리 설정 (컨테이너 내부의 /app이 ai 폴더의 내용에 해당함)
WORKDIR /app

# 모든 프로젝트 파일 복사 (.env, run_training.sh, requirements.txt 등 포함)
COPY . /app

# 1. dos2unix 설치 및 run_training.sh 변환 (CRLF -> LF 강제 변환)
RUN apt-get update && \
    apt-get install -y dos2unix && \
    dos2unix run_training.sh && \
    rm -rf /var/lib/apt/lists/*

# 2. 필수 라이브러리 설치 (python-dotenv가 requirements.txt에 없다고 가정하고 설치)
RUN pip install -r requirements.txt

# 3. 쉘 스크립트에 실행 권한 부여
RUN chmod +x run_training.sh

# 4. 포트 노출
EXPOSE 8000

# 5. CMD 명령 방식 변경 (Exec format error 우회 및 안정성 확보)
# /bin/bash 쉘을 명시적으로 사용하여 스크립트를 실행합니다.
CMD ["/bin/bash", "./run_training.sh"]