# 1단계: 빌드 환경
FROM gradle:8.5-jdk17 AS java-build
WORKDIR /app

# backend 전체 복사
COPY . /app

# gradlew 실행 권한 부여
RUN chmod +x gradlew

# 빌드 실행
# ./gradlew clean build 명령어는 /app 디렉토리에서 실행됩니다.
RUN ./gradlew clean build -x test

# 2단계: 실행 환경
FROM openjdk:17.0.2-jdk-slim
WORKDIR /app

# 빌드된 JAR 파일 복사
COPY --from=java-build /app/build/libs/*.jar app.jar

EXPOSE 8080
ENV PORT 8080

CMD ["java", "-jar", "app.jar"]
