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
                        .requestMatchers(HttpMethod.GET, "/api/stores/**").permitAll()
                        .requestMatchers(HttpMethod.GET, "/api/stores/*/reviews/**").permitAll()
                        .requestMatchers(HttpMethod.GET, "/api/posts/**").permitAll()

                        // ✅ (추가) 길찾기: 프론트에서 POST로 호출하니까 명시적으로 열기
                        .requestMatchers(HttpMethod.GET, "/api/routes/**").permitAll()
                        .requestMatchers(HttpMethod.POST, "/api/routes/**").permitAll()

                        // ✅ (추가) 스프링에서 요약 API를 제공한다면 열기 (없으면 삭제해도 됨)
                        .requestMatchers(HttpMethod.GET, "/api/stores/*/summary/**").permitAll()

                        // ✅ (선택) 로그인 안 했을 때 즐겨찾기 GET 호출로 403 나는 거 싫으면
                        // ⚠️ 단, 즐겨찾기가 "유저 개인 데이터"면 공개로 열면 안 됨!
                        .requestMatchers(HttpMethod.GET, "/api/favorites/**").permitAll()

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

                        // 리뷰 작성/수정/삭제는 로그인 필요 (GET은 위에서 permitAll)
                        .requestMatchers("/api/stores/*/reviews/**").authenticated()

                        // 관리자용 API
                        .requestMatchers("/api/admin/**").hasRole("ADMIN")

                        // 프로필, 로그아웃, 탈퇴
                        .requestMatchers(
                                "/api/auth/logout",
                                "/api/profile",
                                "/api/auth/withdrawal"
                        ).authenticated()

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
