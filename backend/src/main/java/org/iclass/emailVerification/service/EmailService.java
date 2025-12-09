package org.iclass.emailVerification.service;

import java.time.LocalDateTime;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.mail.javamail.JavaMailSender;
import org.springframework.mail.javamail.MimeMessageHelper;
import org.springframework.stereotype.Service;
import org.springframework.beans.factory.annotation.Value;
import jakarta.mail.internet.MimeMessage;
import lombok.extern.slf4j.Slf4j;

@Service
@Slf4j
public class EmailService {

    @Autowired
    private JavaMailSender mailSender;

    @Value("${spring.mail.username}")
    private String fromEmail;
    
    // 보안: 코드와 만료 시간을 함께 저장하기 위한 내부 클래스
    private static class VerificationInfo {
        String code;
        LocalDateTime expiry;
        VerificationInfo(String code) {
            this.code = code;
            this.expiry = LocalDateTime.now().plusMinutes(5); // 5분 만료
        }
    }

    private final Map<String, VerificationInfo> verificationCache = new ConcurrentHashMap<>();
    
    public void sendCodeToEmail(String email, String purpose) {
        String verificationCode = generateVerificationCode();
        // 보안: 만료 정보와 함께 캐시 저장
        verificationCache.put(email + "_" + purpose, new VerificationInfo(verificationCode));
        
        boolean success = sendVerificationEmail(email, verificationCode);
        if (!success) throw new IllegalStateException("이메일 발송에 실패했습니다.");
    }
    
    public boolean verifyEmailCode(String email, String code, String purpose) {
        String key = email + "_" + purpose;
        VerificationInfo info = verificationCache.get(key);

        if (info == null) return false;

        // 보안 1: 시간 만료 체크
        if (LocalDateTime.now().isAfter(info.expiry)) {
            verificationCache.remove(key);
            log.warn("인증 코드 만료: {}", email);
            return false;
        }

        // 보안 2: 코드 일치 여부 확인
        boolean isMatch = info.code.equals(code);
        
        // 보안 3: 검증 시도 후 즉시 파기 (성공/실패 관계없이 재사용 방지)
        verificationCache.remove(key);
        
        return isMatch;
    }

    public String generateVerificationCode() {
        return String.format("%06d", (int)(Math.random() * 1000000));
    }

    public boolean sendVerificationEmail(String email, String verificationCode) {
        try {
            MimeMessage message = mailSender.createMimeMessage();
            MimeMessageHelper helper = new MimeMessageHelper(message, true, "UTF-8");
            helper.setFrom(fromEmail);
            helper.setTo(email);
            helper.setSubject("[디쉬인사이드] 이메일 인증번호입니다");
            helper.setText(createVerificationCodeHtml(verificationCode), true);
            mailSender.send(message);
            return true;
        } catch (Exception e) {
            log.error("이메일 발송 실패: {}", e.getMessage());
            return false;
        }
    }

    private String createVerificationCodeHtml(String verificationCode) {
        return """
            <div style="font-family: Arial, sans-serif; padding: 20px;">
                <h2>이메일 인증번호</h2>
                <p>회원가입을 위해 아래의 6자리 인증번호를 입력해주세요.</p>
                <div style="background: #f4f4f4; padding: 15px; font-size: 24px; font-weight: bold; letter-spacing: 5px;">
                    %s
                </div>
                <p style="color: red;">* 이 인증번호는 5분 동안만 유효합니다.</p>
            </div>
            """.formatted(verificationCode);
    }
}