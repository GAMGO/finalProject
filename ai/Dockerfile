# Dockerfile (ai 루트 폴더에 위치)

# 베이스 이미지: Python 3.11 사용
FROM python:3.11-slim

# 작업 디렉토리 설정 (컨테이너 내부의 /app이 ai 폴더의 내용에 해당함)
WORKDIR /app

# 모든 프로젝트 파일 복사
# ai 폴더 내의 모든 내용을 /app으로 복사합니다.
COPY . /app

# 쉘 스크립트에 실행 권한 부여
RUN chmod +x run_training.sh

# 포트 노출
EXPOSE 8000

# Railway 환경 변수 관리를 위해 .env 파일 대신 Dockerfile 내에서 설치
# (프로젝트에 .env 파일이 있다면 COPY . /app 에서 이미 포함됩니다.)
RUN pip install python-dotenv

# 컨테이너 시작 시 run_training.sh 실행
# 이 스크립트가 환경 설정 수정, 학습, 최종 Uvicorn 실행까지 모두 처리합니다.
CMD ["./run_training.sh"]