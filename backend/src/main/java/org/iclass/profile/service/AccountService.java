// src/main/java/org/iclass/profile/service/AccountService.java
package org.iclass.profile.service;

import java.time.Duration;
import java.time.LocalDateTime;
import java.util.Comparator;
import java.util.List;

import org.iclass.customer.entity.CustomersEntity;
import org.iclass.customer.repository.CustomersRepository;
import org.iclass.customer.service.CustomersService;
import org.iclass.emailVerification.service.EmailService;
import org.iclass.profile.dto.AccountInfoDto;
import org.iclass.profile.dto.AccountUpdateRequest;
import org.iclass.profile.dto.PasswordResetRequest;
import org.iclass.recovery.entity.Recovery;
import org.iclass.recovery.repository.RecoveryRepository;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.security.core.userdetails.UsernameNotFoundException;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import lombok.RequiredArgsConstructor;

@RequiredArgsConstructor
@Service
public class AccountService {

    private final CustomersRepository customersRepository;
    private final CustomersService customersService;
    private final EmailService emailService;
    private final RecoveryRepository recoveryRepository;

    private static final Duration RECOVERY_CODE_TTL = Duration.ofMinutes(5);

    // 현재 로그인한 고객 엔티티 가져오기
    private CustomersEntity getCurrentCustomer(UserDetails userDetails) {
        if (userDetails == null) {
            throw new UsernameNotFoundException("인증된 사용자가 없습니다.");
        }
        String username = userDetails.getUsername();   // = customer_id
        return customersRepository.findById(username)
                .orElseThrow(() -> new UsernameNotFoundException("사용자를 찾을 수 없습니다 : " + username));
    }

    // ----------------------------------------------------------
    // 1) 회원정보 조회
    // ----------------------------------------------------------
    @Transactional(readOnly = true)
    public AccountInfoDto getMyAccount(UserDetails userDetails) {
        CustomersEntity me = getCurrentCustomer(userDetails);

        return AccountInfoDto.builder()
                .id(me.getId())
                .email(me.getEmail())
                .birth(me.getBirth())
                .age(me.getAge())
                .gender(me.getGender())
                .address(me.getAddress())
                .emailVerified(Boolean.TRUE.equals(me.getEmailVerified()))
                .build();
    }

    // ----------------------------------------------------------
    // 2) 회원정보 수정 (birth/age/gender/address 정도만)
    // ----------------------------------------------------------
    @Transactional
    public AccountInfoDto updateMyAccount(UserDetails userDetails, AccountUpdateRequest req) {
        CustomersEntity me = getCurrentCustomer(userDetails);

        if (req.getBirth() != null) {
            me.setBirth(req.getBirth());
        }
        if (req.getAge() != null) {
            me.setAge(req.getAge());
        }
        if (req.getGender() != null || me.getGender() != null) {
            // null 로 보내면 비공개로 변경하는 것도 허용
            me.setGender(req.getGender());
        }
        if (req.getAddress() != null) {
            me.setAddress(req.getAddress());
        }

        CustomersEntity saved = customersRepository.save(me);

        return AccountInfoDto.builder()
                .id(saved.getId())
                .email(saved.getEmail())
                .birth(saved.getBirth())
                .age(saved.getAge())
                .gender(saved.getGender())
                .address(saved.getAddress())
                .emailVerified(Boolean.TRUE.equals(saved.getEmailVerified()))
                .build();
    }

    // ----------------------------------------------------------
    // 3) 비밀번호 변경용 코드 이메일 전송
    //    (기존 EmailService / Recovery 재활용)
    // ----------------------------------------------------------
    @Transactional
    public void sendPasswordResetCode(UserDetails userDetails) {
        CustomersEntity me = getCurrentCustomer(userDetails);

        if (me.getEmail() == null || me.getEmail().isBlank()) {
            throw new IllegalStateException("등록된 이메일이 없어 비밀번호 변경 코드를 보낼 수 없습니다.");
        }

        // 기존 로직 재사용: 6자리 랜덤 코드 생성
        String code = emailService.generateVerificationCode();

        // Recovery 엔티티에 저장 (가장 최근 것만 쓰도록 updatedAt 갱신)
        Recovery recovery = Recovery.builder()
                .customer(me)
                .emailVerifiedCode(code)
                .updatedAt(LocalDateTime.now())
                .build();

        recoveryRepository.save(recovery);

        // 이메일 발송
        boolean sent = emailService.sendVerificationEmail(me.getEmail(), code);
        if (!sent) {
            throw new IllegalStateException("비밀번호 변경용 인증 이메일 발송에 실패했습니다.");
        }
    }

    // ----------------------------------------------------------
    // 4) 코드 검증 + 비밀번호 변경
    // ----------------------------------------------------------
    @Transactional
    public void resetPassword(UserDetails userDetails, PasswordResetRequest req) {
        CustomersEntity me = getCurrentCustomer(userDetails);

        // 1) 이 유저의 가장 최근 Recovery 레코드를 찾는다
        List<Recovery> all = recoveryRepository.findAll(); // 규모가 크지 않다고 가정
        Recovery latest = all.stream()
                .filter(r -> r.getCustomer() != null
                        && r.getCustomer().getIdx().equals(me.getIdx()))
                .max(Comparator.comparing(Recovery::getUpdatedAt))
                .orElseThrow(() -> new IllegalStateException("발급된 비밀번호 변경 코드가 없습니다."));

        // 2) 코드 일치 여부
        if (latest.getEmailVerifiedCode() == null
                || !latest.getEmailVerifiedCode().equals(req.getCode())) {
            throw new IllegalArgumentException("인증 코드가 올바르지 않습니다.");
        }

        // 3) 유효시간(5분) 체크
        LocalDateTime updatedAt = latest.getUpdatedAt();
        if (updatedAt == null ||
                updatedAt.isBefore(LocalDateTime.now().minus(RECOVERY_CODE_TTL))) {
            throw new IllegalStateException("인증 코드가 만료되었습니다. 다시 요청해 주세요.");
        }

        // 4) 기존 CustomersService의 비밀번호 변경 유틸 재사용
        customersService.updatePassword(me.getId(), req.getNewPassword());
    }
}
