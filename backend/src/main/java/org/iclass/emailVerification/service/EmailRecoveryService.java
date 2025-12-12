package org.iclass.emailVerification.service;

import jakarta.mail.internet.MimeMessage;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.mail.javamail.JavaMailSender;
import org.springframework.mail.javamail.MimeMessageHelper;
import org.springframework.stereotype.Service;

@Service
@Slf4j
public class EmailRecoveryService {

    @Autowired
    private JavaMailSender mailSender;

    @Value("${spring.mail.username}")
    private String fromEmail;

    @Value("${app.frontend.url}")
    private String frontendUrl; // 예: http://localhost:5173

    @Value("${VITE_LOCAL_BASE_URL}") 
    private String backendApiUrl;
    /**
     * 회원 탈퇴 완료 안내 및 복구 코드 발송
     * @param email 수신자 이메일
     * @param recoveryCode 복구 인증 코드
     */
    public void sendWithdrawalNotification(String email, String recoveryCode) {
        try {
            MimeMessage message = mailSender.createMimeMessage();
            MimeMessageHelper helper = new MimeMessageHelper(message, true, "UTF-8");

            helper.setFrom(fromEmail);
            helper.setTo(email);
            helper.setSubject("[디쉬인사이드] 회원 탈퇴 처리가 완료되었습니다.");

            String recoveryLink = backendApiUrl + "/api/auth/recovery";
            String htmlContent = createWithdrawalHtml(recoveryCode, recoveryLink);
            
            helper.setText(htmlContent, true);
            mailSender.send(message);
            
            log.info("회원 탈퇴 알림 이메일 발송 완료: {}", email);
        } catch (Exception e) {
            log.error("회원 탈퇴 이메일 발송 실패: {}", e.getMessage());
        }
    }
     //탈퇴 알림 HTML 템플릿 생성
    private String createWithdrawalHtml(String recoveryCode, String recoveryLink) {
        return """
                <!DOCTYPE html>
                <html>
                <body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px;">
                    <h2 style="color: #d9534f;">회원 탈퇴가 완료되었습니다.</h2>
                    <p>안녕하세요. 그동안 저희 서비스를 이용해 주셔서 감사합니다.</p>
                    <p>본 요청이 본인의 의사가 아니거나, 계정을 복구하고 싶으시다면 아래의 복구 코드를 사용해 주세요.</p>
                    <div style="background-color: #f8f9fa; padding: 20px; text-align: center; border-radius: 8px;">
                        <p style="margin: 0; font-size: 14px; color: #666;">계정 복구 인증 코드</p>
                        <h1 style="color: #007bff; letter-spacing: 5px;">%s</h1>
                    </div>
                    <div style="text-align: center; margin-top: 30px;">
                        <a href="%s" style="display: inline-block; padding: 12px 25px; background-color: #78266A; color: #ffffff !important; text-decoration: none; border-radius: 25px; font-size: 16px; font-weight: bold; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                            계정 복구 페이지로 이동
                        </a>
                    </div>
                    <p style="font-size: 13px; color: #999;">* 이 코드는 보안을 위해 일정 시간 동안만 유효합니다.</p>
                    <hr>
                    <p style="font-size: 12px; color: #ccc;">본인이 탈퇴를 진행하지 않았다면 즉시 고객센터로 문의해 주세요.</p>
                </body>
                </html>
                """.formatted(recoveryCode, recoveryLink);
    }
}