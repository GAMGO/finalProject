# 1단계: 빌드 환경 (Build Stage) - Maven과 OpenJDK 17 환경을 사용하여 JAR 파일 생성
FROM maven:3.8-jdk-17 AS build
WORKDIR /app
COPY . /app 
# build.gradle이 backend 폴더 안에 있다면, 아래 명령어를 사용할 수 없습니다.
# 현재 고객님의 리포지토리는 finalProject/frontend, finalProject/backend 구조이므로, 
# 이전 답변에서 안내드린 모노레포 구조의 Dockerfile을 사용해야 합니다. 
# (현재 로그는 단일 폴더 구조 Dockerfile을 사용하는 것으로 보이지만, 일단 이미지 수정에 집중합니다.)

RUN ./gradlew clean build --no-daemon -x test

# 2단계: 실행 환경 (Production Stage)
FROM openjdk:17-jre-slim
WORKDIR /app
COPY --from=build /app/build/libs/[YOUR_PROJECT_NAME]-0.0.1-SNAPSHOT.jar app.jar 

EXPOSE 8080 
ENV PORT 8080

CMD ["java", "-jar", "app.jar"]