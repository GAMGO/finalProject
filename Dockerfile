# 1단계: 빌드 환경 (Gradle + JDK 17)
FROM gradle:8.5-jdk17 AS build

WORKDIR /app

# gradlew와 backend 폴더를 명시적으로 복사
COPY backend/gradlew /app/gradlew
COPY backend /app/backend

# gradlew 실행 권한 부여
RUN chmod +x gradlew

# backend 디렉토리로 이동
WORKDIR /app/backend

# ../gradlew로 빌드 실행
RUN ../gradlew clean build --no-daemon -x test

# 2단계: 실행 환경 (가벼운 JDK 이미지)
FROM openjdk:17.0.2-jdk
WORKDIR /app

# 빌드된 JAR 파일 복사
COPY --from=build /app/backend/build/libs/finalProject-0.0.1-SNAPSHOT.jar app.jar

EXPOSE 8080
ENV PORT 8080

CMD ["java", "-jar", "app.jar"]
