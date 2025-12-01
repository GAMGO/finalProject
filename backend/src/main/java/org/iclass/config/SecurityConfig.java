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
                        "/api/auth/**", // 로그인/회원가입 등
                        "/error", // 스프링 기본 에러 엔드포인트
                        "/api/recover/**", // >>> [ADDED] 비밀번호 복구(시작/검증/재설정) 공개 엔드포인트
                        "/api/food/**",
                        "/api/email/**" // 이메일 인증 관련 엔드포인트
        };

        @Bean
        public SecurityFilterChain securityFilterChain(HttpSecurity http,
                        JwtAuthenticationFilter jwtAuthenticationFilter) throws Exception {
                http
                                // corsConfig 빈에서 가져온 설정을 직접 사용
                                .cors(cors -> cors.configurationSource(corsConfig.corsConfigurationSource()))
                                .csrf(csrf -> csrf.disable())
                                .sessionManagement(sm -> sm.sessionCreationPolicy(SessionCreationPolicy.STATELESS))
                                .addFilterBefore(jwtAuthenticationFilter, UsernamePasswordAuthenticationFilter.class)
                                .authorizeHttpRequests(auth -> auth
                                                .requestMatchers(SWAGGER_WHITELIST).permitAll()
                                                .requestMatchers(HttpMethod.OPTIONS, "/**").permitAll()
                                                .requestMatchers(PUBLIC_WHITELIST).permitAll()
                                                .requestMatchers("/api/email/**","/api/stores").permitAll() // 이메일 인증, 가게목록 명시적 허용
                                                .requestMatchers(HttpMethod.POST,"/api/auth/login","/api/recover/send-code","/api/email/verify","/api/recover/reset","/api/recover/find-id")
                                                .permitAll()
                                                .requestMatchers("/api/auth/logout","/api/posts","/api/favorites","/api/profile")
                                                .authenticated()
                                                .anyRequest().authenticated())
                                                
                                // 폼/베이직 로그인 비활성
                                .httpBasic(b -> b.disable())
                                .formLogin(f -> f.disable())
                                .logout(l -> l
                                    // 기본 로그아웃 URL을 사용하지 않도록 처리
                                    .logoutUrl("/non-existent-logout") 
                                    // 로그아웃 성공 시 리다이렉션 방지 (REST API에서는 중요)
                                    .logoutSuccessHandler((request, response, authentication) -> {})
                                    // 세션 기반이 아니므로 세션 무효화는 필요 없음
                                    .invalidateHttpSession(false) 
                                    .clearAuthentication(false)
                                    .permitAll()); // 로그아웃 처리는 PUBLIC_WHITELIST에서 이미 허용됨

                return http.build();
        }

        @Bean
        public PasswordEncoder passwordEncoder() {
                return new BCryptPasswordEncoder();
        }

        @Bean
        public AuthenticationManager authenticationManager(AuthenticationConfiguration authenticationConfiguration)
                        throws Exception {
                return authenticationConfiguration.getAuthenticationManager();
        }

}