package org.iclass.config;

import lombok.RequiredArgsConstructor;
import org.iclass.security.JwtAuthenticationFilter;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.http.HttpMethod;
import org.springframework.scheduling.annotation.EnableScheduling;
import org.springframework.security.authentication.AuthenticationManager;
import org.springframework.security.config.annotation.authentication.configuration.AuthenticationConfiguration;
import org.springframework.security.config.http.SessionCreationPolicy;
import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.security.web.SecurityFilterChain;
import org.springframework.security.web.authentication.UsernamePasswordAuthenticationFilter;
import org.springframework.security.config.annotation.web.builders.HttpSecurity;

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

    // 완전 공개 허용 (개발용 / 인증 불필요)
    private static final String[] PUBLIC_WHITELIST = {
            "/api/auth/**",       // 로그인/회원가입 (아래에서 일부 POST만 따로 허용 재정의)
            "/error",
            "/api/recover/**",
            "/api/food/**",
            "/api/email/**"
    };

    @Bean
    public SecurityFilterChain securityFilterChain(
            HttpSecurity http,
            JwtAuthenticationFilter jwtAuthenticationFilter) throws Exception {

        http
                .cors(cors -> cors.configurationSource(corsConfig.corsConfigurationSource()))
                .csrf(csrf -> csrf.disable())
                .sessionManagement(sm -> sm.sessionCreationPolicy(SessionCreationPolicy.STATELESS))
                .addFilterBefore(jwtAuthenticationFilter, UsernamePasswordAuthenticationFilter.class)
                .authorizeHttpRequests(auth -> auth
                        // ===== 공통 허용 =====
                        .requestMatchers(SWAGGER_WHITELIST).permitAll()
                        .requestMatchers(HttpMethod.OPTIONS, "/**").permitAll()
                        .requestMatchers(PUBLIC_WHITELIST).permitAll()

                        // ====== 누구나 볼 수 있는 GET API ======
                        // 가게 목록/상세, 리뷰 조회는 비로그인도 가능
                        .requestMatchers(HttpMethod.GET, "/api/stores/**").permitAll()
                        // /api/stores/{storeIdx}/reviews, /with-stats 등 GET 전부 허용
                        .requestMatchers(HttpMethod.GET, "/api/stores/*/reviews/**").permitAll()
                        // 게시글, 경로 조회도 GET은 허용
                        .requestMatchers(HttpMethod.GET, "/api/posts/**").permitAll()
                        .requestMatchers("/api/routes/**").permitAll()

                        // ====== 인증 없이 접근해야 하는 POST ======
                        .requestMatchers(HttpMethod.POST,
                                "/api/auth/login",
                                "/api/auth/refresh",       // 토큰 재발급
                                "/api/recover/send-code",
                                "/api/recover/reset",
                                "/api/recover/find-id"
                        ).permitAll()

                        // ===== 로그인 필수 영역 =====
                        // 찜(즐겨찾기): 목록/추가/수정/삭제 모두 로그인 필요
                        .requestMatchers("/api/favorites/**").authenticated()

                        // 리뷰 작성/수정/삭제는 로그인 필요
                        // (GET 은 위에서 이미 permitAll 처리)
                        .requestMatchers("/api/stores/*/reviews/**").authenticated()

                        // 프로필, 로그아웃, 탈퇴 등
                        .requestMatchers(
                                "/api/auth/logout",
                                "/api/profile",
                                "/api/auth/withdrawal"
                        ).authenticated()

                        // 그 외 전부 로그인 필요
                        .anyRequest().authenticated()
                )
                // 폼/베이직 로그인 비활성
                .httpBasic(b -> b.disable())
                .formLogin(f -> f.disable())
                .logout(l -> l
                        // 기본 로그아웃 URL 비활성 (컨트롤러에서 직접 처리한다고 가정)
                        .logoutUrl("/non-existent-logout")
                        .logoutSuccessHandler((request, response, authentication) -> {
                        })
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
            AuthenticationConfiguration authenticationConfiguration) throws Exception {
        return authenticationConfiguration.getAuthenticationManager();
    }
}
