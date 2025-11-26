# 1단계: 빌드 환경
FROM gradle:8.5-jdk17 AS build
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

# 2단계: 실행 환경
FROM openjdk:17.0.2-jdk-slim
WORKDIR /app

# 빌드된 JAR 파일 복사
COPY --from=build /app/backend/build/libs/app.jar app.jar

EXPOSE 8080
ENV PORT 8080

CMD ["java", "-jar", "app.jar"]
