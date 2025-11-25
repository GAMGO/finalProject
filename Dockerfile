# 1단계: 빌드 환경 (Build Stage) - Gradle과 OpenJDK 17 환경을 사용하여 JAR 파일 생성
FROM gradle:8.5-jdk17 AS build
WORKDIR /app
COPY . /app 
WORKDIR /app/backend 

# [필수 추가]: gradlew 파일에 실행 권한 부여
RUN chmod +x gradlew  # <--- 이 줄을 추가합니다.
# 현재 리포지토리가 finalProject/backend 구조라면 아래 명령어를 사용해야 합니다.
# 만약 backend 폴더 전체가 소스코드라면, 아래와 같이 폴더 경로를 명시합니다.
# Dockerfile이 finalProject 루트에 있다면, 모든 파일을 복사합니다.
COPY . /app 
# backend 폴더로 작업 디렉터리를 변경합니다.
WORKDIR /app/backend 

# 빌드 실행
RUN ./gradlew clean build --no-daemon -x test

# 2단계: 실행 환경 (Production Stage)
FROM openjdk:17.0.2-jdk
WORKDIR /app

# JAR 파일 복사 경로 (backend 폴더 내의 build/libs에서 복사)
COPY --from=build /app/backend/build/libs/[YOUR_PROJECT_NAME]-0.0.1-SNAPSHOT.jar app.jar 

EXPOSE 8080 
ENV PORT 8080

CMD ["java", "-jar", "app.jar"]