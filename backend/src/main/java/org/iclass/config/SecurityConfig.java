package org.iclass.config;

import org.iclass.security.JwtAuthenticationFilter;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.http.HttpMethod;
import org.springframework.scheduling.annotation.EnableScheduling;
import org.springframework.security.authentication.AuthenticationManager;
import org.springframework.security.config.annotation.authentication.configuration.AuthenticationConfiguration;
import org.springframework.security.config.annotation.web.builders.HttpSecurity;
import org.springframework.security.config.http.SessionCreationPolicy;
import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.security.web.SecurityFilterChain;
import org.springframework.security.web.authentication.UsernamePasswordAuthenticationFilter;

import lombok.RequiredArgsConstructor;

@RequiredArgsConstructor
@Configuration
@EnableScheduling
public class SecurityConfig {

    private final CorsConfig corsConfig;

    // swagger 문서 접근 허용 목록
    private static final String[] SWAGGER_WHITELIST = {
            "/v3/api-docs/**",
            "/swagger-ui/**",
            "/swagger-ui.html"
    };

    // 공개 허용 (개발용)
    private static final String[] PUBLIC_WHITELIST = {
            "/api/auth/**",     // 로그인/회원가입 등
            "/error",           // 스프링 기본 에러 엔드포인트
            "/api/recover/**",  // 비밀번호 복구(시작/검증/재설정)
            "/api/food/**",
            "/api/email/**"     // 이메일 인증 관련 엔드포인트
    };

    @Bean
    public SecurityFilterChain securityFilterChain(
            HttpSecurity http,
            JwtAuthenticationFilter jwtAuthenticationFilter
    ) throws Exception {

        http
            .cors(cors -> cors.configurationSource(corsConfig.corsConfigurationSource()))
            .csrf(csrf -> csrf.disable())
            .sessionManagement(sm -> sm.sessionCreationPolicy(SessionCreationPolicy.STATELESS))
            .addFilterBefore(jwtAuthenticationFilter, UsernamePasswordAuthenticationFilter.class)
            .authorizeHttpRequests(auth -> auth
                    // swagger, preflight, 공개 엔드포인트
                    .requestMatchers(SWAGGER_WHITELIST).permitAll()
                    .requestMatchers(HttpMethod.OPTIONS, "/**").permitAll()
                    .requestMatchers(PUBLIC_WHITELIST).permitAll()

                    // 이메일 인증, 가게 목록 (POST 포함 전체) 허용
                    .requestMatchers("/api/email/**", "/api/stores/**","/api/stores/{storeIdx}/reviews").permitAll()

                //     // ====== 🔥 리뷰 관련 추가 ======
                //     // 리뷰 조회는 누구나 (with-stats 포함)
                //     .requestMatchers(
                //             HttpMethod.GET,
                //             "/api/stores/*/reviews/**"
                //     ).permitAll()

                //     // 리뷰 작성/수정/삭제는 로그인 필요
                //     .requestMatchers(
                //             HttpMethod.POST,
                //             "/api/stores/*/reviews"
                //     ).authenticated()
                //     .requestMatchers(
                //             HttpMethod.PUT,
                //             "/api/stores/*/reviews/*"
                //     ).authenticated()
                //     .requestMatchers(
                //             HttpMethod.DELETE,
                //             "/api/stores/*/reviews/*"
                //     ).authenticated()
                //     // ====== 여기까지 리뷰 ======

                    // 로그인/비번찾기 POST는 허용
                    .requestMatchers(
                            HttpMethod.POST,
                            "/api/auth/login",
                            "/api/recover/send-code",
                            "/api/email/verify",
                            "/api/recover/reset",
                            "/api/recover/find-id",
                            "/api/stores/{storeIdx}/reviews"
                    ).permitAll()

                    // 나머지 일부 API는 로그인 필요
                    .requestMatchers(
                            "/api/auth/logout",
                            "/api/posts",
                            "/api/favorites",
                            "/api/profile"
                    ).authenticated()

                    // 그 외 전부 인증 필요
                    .anyRequest().authenticated()
            )
            // 폼/베이직 로그인 비활성
            .httpBasic(b -> b.disable())
            .formLogin(f -> f.disable())
            .logout(l -> l
                    .logoutUrl("/non-existent-logout")
                    .logoutSuccessHandler((request, response, authentication) -> {})
                    .invalidateHttpSession(false)
                    .clearAuthentication(false)
                    .permitAll()
            );

        return http.build();
    }

    @Bean
    public PasswordEncoder passwordEncoder() {
        return new BCryptPasswordEncoder();
    }

    @Bean
    public AuthenticationManager authenticationManager(
            AuthenticationConfiguration authenticationConfiguration
    ) throws Exception {
        return authenticationConfiguration.getAuthenticationManager();
    }
}
