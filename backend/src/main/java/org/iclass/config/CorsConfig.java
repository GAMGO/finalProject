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
        c.addAllowedOriginPattern("https://merry-bunny-6557aa.netlify.app");
        c.addAllowedOriginPattern("https://api.dishinside.shop");
        c.addAllowedOriginPattern("http://localhost:5173");  // Vite dev 서버
        c.addAllowedOriginPattern("http://localhost:8080");  // 사용 시
        c.addAllowedOriginPattern("http://127.0.0.1:8000");  // 사용 시
        c.setAllowedMethods(List.of("GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"));
        c.addAllowedHeader("*");
        c.setAllowedHeaders(List.of("Content-Type", "Authorization"));
        c.setAllowCredentials(true);
        UrlBasedCorsConfigurationSource s = new UrlBasedCorsConfigurationSource();
        s.registerCorsConfiguration("/**", c);
        return s;
    }
}
