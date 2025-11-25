# 1단계: 빌드 환경 (Build Stage) - Gradle과 OpenJDK 17 환경을 사용하여 JAR 파일 생성
FROM gradle:8.5-jdk17 AS build

# 작업 디렉토리를 /app으로 설정합니다. (여기가 리포지토리 루트가 됩니다)
WORKDIR /app
# 로컬 파일 전체를 Docker 이미지의 /app 디렉토리로 복사합니다.
# gradlew 파일은 이 단계에서 /app/gradlew 에 복사됩니다.
COPY . /app 

# --- [권한 문제 해결 영역] ---
# gradlew 파일은 /app에 있으므로, 작업 디렉토리가 /app인 상태에서 권한을 부여합니다.
RUN chmod +x gradlew

# build.gradle이 있는 백엔드 서브 프로젝트 디렉토리로 작업 디렉토리를 변경합니다.
WORKDIR /app/backend 

# 빌드 실행
# 상위 디렉토리(/app)에 있는 gradlew를 참조하여 빌드를 실행합니다.
RUN ../gradlew clean build --no-daemon -x test

# 2단계: 실행 환경 (Production Stage)
# 더 가벼운 JDK slim 이미지를 권장합니다.
FROM openjdk:17.0.2-jdk
WORKDIR /app

# JAR 파일 복사 경로 (backend 폴더 내의 build/libs에서 복사)
# [YOUR_PROJECT_NAME] 부분을 실제 프로젝트 이름으로 반드시 교체해 주세요!
COPY --from=build /app/backend/build/libs/finalProject-0.0.1-SNAPSHOT.jar app.jar 

EXPOSE 8080 
ENV PORT 8080

CMD ["java", "-jar", "app.jar"]