package org.iclass.config;

import java.util.List;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.cors.CorsConfiguration;
import org.springframework.web.cors.CorsConfigurationSource;
import org.springframework.web.cors.UrlBasedCorsConfigurationSource;

@Configuration
public class CorsConfig {
    @Bean
    public CorsConfigurationSource corsConfigurationSource() {
        CorsConfiguration c = new CorsConfiguration();
        c.addAllowedOriginPattern("https://dishinside.shop");
        c.addAllowedOriginPattern("https://api.dishinside.shop");
        c.addAllowedOriginPattern("http://localhost:5173");  // Vite dev 서버
        c.addAllowedOriginPattern("http://localhost:3000");  // 사용 시
        c.addAllowedOriginPattern("http://localhost:*");     // 모든 로컬 포트 허용
        c.setAllowedMethods(List.of("GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"));
        c.addAllowedHeader("*");
        c.setAllowedHeaders(List.of("Content-Type", "Authorization"));
        c.setAllowCredentials(true);
        UrlBasedCorsConfigurationSource s = new UrlBasedCorsConfigurationSource();
        s.registerCorsConfiguration("/**", c);
        return s;
    }
}
