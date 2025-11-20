package org.iclass.finalproject.config;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;
import org.springframework.security.crypto.password.PasswordEncoder;

@Configuration
public class SecurityBeans {

    @Bean
    public PasswordEncoder passwordEncoder() {
        return new BCryptPasswordEncoder();
    }
}

/*
 * [파일 설명]
 * - Security 관련 Bean 등록용 설정 클래스.
 * - 현재는 PasswordEncoder(BCrypt)만 등록해서 회원 비밀번호 암호화에 사용.
 * - 추후 JWT 필터/보안설정이 추가되더라도 공통으로 사용하는 Bean은 여기에서 관리.
 */
