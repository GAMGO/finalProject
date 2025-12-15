package org.iclass.config;

import lombok.RequiredArgsConstructor;
import org.iclass.security.JwtAuthenticationFilter;
import org.springframework.context.annotation.*;
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

    private static final String[] SWAGGER_WHITELIST = {
            "/v3/api-docs/**",
            "/swagger-ui/**",
            "/swagger-ui.html"
    };

    private static final String[] PUBLIC_WHITELIST = {
            "/api/auth/**",
            "/error",
            "/api/recover/**",
            "/api/food/**",
            "/api/email/**",
            "/api/recovery/**",
            "/api/routes/**"
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
                        .requestMatchers(HttpMethod.GET, "/api/favorites/**","/api/stores/**","/api/stores/*/reviews/**","/api/posts/**","/api/routes/**","/api/stores/*/summary/**").permitAll()
                        .requestMatchers(HttpMethod.POST,"/api/auth/login","/api/auth/refresh", "/api/routes/**").permitAll()// 토큰 재발급
                        // ====== 인증 없이 접근해야 하는 POST ======
                        .requestMatchers(HttpMethod.POST,
                                "/api/auth/login",
                                "/api/auth/refresh",
                                "/api/recover/send-code",
                                "/api/recover/reset",
                                "/api/recover/find-id",
                                "/api/stores",
                                "/api/stores/*/update-request",
                                "/api/stores/*/delete-request"
                        ).permitAll()
                        // ===== 로그인 필수 영역 =====
                        // ✅ (주의) 위에서 GET /api/favorites/** permitAll을 해놨으면
                        // 여기 authenticated는 "POST/PUT/DELETE"만 막는 효과가 됨 (순서상 OK)
                        .requestMatchers("/api/favorites/**").authenticated()
                        // 리뷰 작성/수정/삭제는 로그인 필요
                        // (GET 은 위에서 이미 permitAll 처리)
                        .requestMatchers("/api/stores/*/reviews/**").authenticated()
                        // 관리자용 API (가게 변경 승인/반려 등)
                        .requestMatchers("/api/admin/**").hasRole("ADMIN")
                        // 프로필, 로그아웃, 탈퇴 등
                        .requestMatchers(
                                "/api/auth/logout",
                                "/api/profile",
                                "/api/auth/withdrawal"
                        ).authenticated()
                        // 그 외 전부 로그인 필요
                        .anyRequest().authenticated()
                )
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
            AuthenticationConfiguration authenticationConfiguration) throws Exception {
        return authenticationConfiguration.getAuthenticationManager();
    }
}
