# Dockerfile (ai 루트 폴더에 위치)

# 베이스 이미지: Python 3.11 사용
FROM python:3.11-slim

# 작업 디렉토리 설정 (컨테이너 내부의 /app이 ai 폴더의 내용에 해당함)
WORKDIR /app

# 모든 프로젝트 파일 복사
# ai 폴더 내의 모든 내용을 /app으로 복사합니다. (.env, run_training.sh 포함)
COPY . /app

# 1. 환경 변수 로딩을 위한 라이브러리 독립 설치 (요청에 따라 먼저 실행)
RUN pip install python-dotenv

# 2. 줄바꿈 문자 문제 해결을 위한 dos2unix 설치 및 변환
# 윈도우(CRLF)에서 생성된 쉘 스크립트를 리눅스(LF) 형식으로 변환합니다.
RUN apt-get update && \
    apt-get install -y dos2unix && \
    dos2unix run_training.sh && \
    # 패키지 캐시 정리 (이미지 크기 최적화)
    rm -rf /var/lib/apt/lists/*

# 3. 쉘 스크립트에 실행 권한 부여
RUN chmod +x run_training.sh

# 4. 포트 노출
EXPOSE 8000

# 5. 컨테이너 시작 시 run_training.sh 실행
# 이 스크립트가 라이브러리 설치, 설정 수정, 학습, 최종 Uvicorn 실행까지 모두 처리합니다.
CMD ["./run_training.sh"]