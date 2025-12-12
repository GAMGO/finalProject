package org.iclass.recovery.controller;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.servlet.view.RedirectView;

@Controller
@RequestMapping("/api/recovery")
public class RecoveryController {

    // application.properties 또는 application.yml에 정의된 프론트엔드 URL 값 (예: http://localhost:5173)
    @Value("${app.frontend.url}")
    private String frontendUrl; 

    @GetMapping("/redirect")
    public RedirectView redirectToFrontend(@RequestParam("token") String token) {
        // 프론트엔드의 RecoveringPage URL에 토큰을 포함하여 리다이렉션 URL 생성
        String redirectUrl = frontendUrl + "/recovery?token=" + token;
        
        RedirectView redirectView = new RedirectView();
        redirectView.setUrl(redirectUrl);
        redirectView.setExposeModelAttributes(false);
        return redirectView;
    }
}