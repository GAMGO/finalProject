# 1단계: 빌드 환경 (Build Stage) - Maven과 OpenJDK 17 환경을 사용하여 JAR 파일 생성
FROM maven:3.8.6-openjdk-17 AS build 

# 작업 디렉터리를 /app으로 설정합니다.
WORKDIR /app 

# 프로젝트의 pom.xml과 소스 코드를 복사합니다.
COPY pom.xml . 
COPY src ./src 

# 애플리케이션을 빌드하여 /target 디렉터리에 JAR 파일을 생성합니다.
# -DskipTests는 테스트 과정을 건너뛰어 빌드 속도를 높입니다.
RUN mvn clean package -DskipTests 

# ----------------------------------------------------------------------
# 2단계: 실행 환경 (Run Stage) - 빌드 결과물만 가져와 최소한의 OpenJDK 17 환경에서 실행
FROM openjdk:17-jdk-slim 

# 1단계에서 빌드된 JAR 파일을 실행 환경의 app.jar로 복사합니다.
# !!!주의!!!: 'your-app-name.jar' 부분을 실제 빌드된 JAR 파일 이름으로 변경해야 합니다.
COPY --from=build /app/target/*.jar app.jar 

# Render는 이 환경 변수(PORT)를 사용해 외부 트래픽을 전달합니다.
# Spring Boot가 이 포트를 사용하도록 설정되어 있어야 합니다. (일반적으로 8080 대신 10000 권장)
ENV PORT 10000 

# 애플리케이션 실행 명령어
CMD ["java", "-jar", "app.jar"]