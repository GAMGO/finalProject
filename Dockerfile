# 1단계: 빌드 환경
FROM gradle:8.5-jdk17 AS java-build
WORKDIR /app

# backend 전체 복사 (gradlew 포함)
COPY backend/gradlew /app/gradlew
COPY backend/gradle /app/gradle
COPY backend /app/backend

# gradlew 실행 권한 부여
RUN chmod +x /app/backend/gradlew

# backend 폴더로 이동
WORKDIR /app/backend

# gradlew 실행 (현재 위치에서 ./gradlew)
RUN ./gradlew clean build --no-daemon -x test
# [2단계: AI/Python 환경 빌드]
# Python 3.11 환경을 설정하고 requirements.txt 종속성을 설치합니다.
FROM python:3.11-slim AS python-build
WORKDIR /app/ai

# requirements.txt 파일 복사 (/ai 폴더에 위치)
COPY /ai/requirements.txt .

# requirements.txt에 명시된 종속성 설치
RUN pip install --no-cache-dir -r requirements.txt

# 필요한 모든 AI 관련 파일 복사 (예: 파이썬 스크립트 등)
COPY /ai/ .

# AI 코드 파일 자체를 복사
COPY /ai /app/ai

# 2단계: 실행 환경
FROM openjdk:17.0.2-jdk-slim
WORKDIR /app

# 빌드된 JAR 파일 복사
COPY --from=java-build /app/backend/build/libs/app.jar app.jar
COPY --from=python-build /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

EXPOSE 8080
ENV PORT 8080

CMD ["java", "-jar", "app.jar"]
