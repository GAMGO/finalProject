package org.iclass.customer.service;

import java.time.LocalDateTime;
import java.util.Optional;

import org.iclass.customer.dto.SignupRequest;
import org.iclass.customer.entity.CustomersEntity;
import org.iclass.customer.repository.CustomersRepository;
import org.iclass.deleteAccount.entity.WithdrawalEntity;
import org.iclass.deleteAccount.repository.WithdrawalRepository;
import org.iclass.emailVerification.service.EmailRecoveryService;
import org.iclass.emailVerification.service.EmailService;

import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.security.core.userdetails.UserDetailsService;
import org.springframework.security.core.userdetails.UsernameNotFoundException;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;

@Slf4j
@RequiredArgsConstructor
@Service
public class CustomersService implements UserDetailsService {

    private final CustomersRepository customersRepository;
    private final PasswordEncoder passwordEncoder;
    private final EmailService emailService;
    private final EmailRecoveryService emailRecoveryService;
    private final WithdrawalRepository withdrawalRepository;

    @Transactional
    public CustomersEntity signup(SignupRequest req) {
        if (customersRepository.existsById(req.getId())) {
            log.warn("[SIGNUP:SERVICE] duplicate id={}", req.getId());
            throw new IllegalStateException("이미 가입된 사용자 ID입니다.");
        }

        CustomersEntity user = CustomersEntity.builder()
                .id(req.getId())
                .password(passwordEncoder.encode(req.getPassword()))
                .age(req.getAge())
                .address(req.getAddress())
                .birth(req.getBirth().toString())
                .email(req.getEmail())
                .emailVerified(true)
                .emailVerificationToken(null)
                .emailVerificationExpires(null)
                .build();

        CustomersEntity savedUser = customersRepository.save(user);
        log.info("[SIGNUP:SERVICE] 새로운 사용자 가입 및 인증 완료: {}", req.getId());
        return savedUser;
    }

    @Transactional
    public void deleteCustomer(String id) {
        CustomersEntity user = customersRepository.findById(id)
                .orElseThrow(() -> new UsernameNotFoundException("사용자를 찾을 수 없습니다: " + id));

        WithdrawalEntity withdrawal = withdrawalRepository.findById(user.getIdx())
                .orElse(WithdrawalEntity.builder().customerIdx(user.getIdx()).build());

        String recoveryCode = emailService.generateVerificationCode();

        withdrawal.setIsDeleted(true);
        withdrawal.setRecoveryToken(recoveryCode);
        withdrawal.setDeletedAt(LocalDateTime.now());

        withdrawalRepository.save(withdrawal);

        user.setRefreshToken(null);
        customersRepository.save(user);

        emailRecoveryService.sendWithdrawalNotification(user.getEmail(), recoveryCode);

        log.info("[WITHDRAWAL:SERVICE] 사용자 소프트 삭제 처리 및 복구 메일 발송: {}", id);
    }

    @Transactional
    public void restoreCustomer(String recoveryToken) {
        WithdrawalEntity withdrawal = withdrawalRepository.findByRecoveryToken(recoveryToken)
                .orElseThrow(() -> new IllegalArgumentException("유효하지 않거나 만료된 복구 토큰입니다."));

        withdrawal.setIsDeleted(false);
        withdrawal.setRecoveryToken(null);
        withdrawalRepository.save(withdrawal);

        log.info("[RESTORE:SERVICE] 계정 복구 성공 (고객 인덱스: {})", withdrawal.getCustomerIdx());
    }

    @Transactional
    public void updatePassword(String customer_id, String newPassword) {
        CustomersEntity user = customersRepository.findById(customer_id)
                .orElseThrow(() -> new UsernameNotFoundException("사용자를 찾을 수 없습니다 : " + customer_id));

        user.setPassword(passwordEncoder.encode(newPassword));
        customersRepository.save(user);

        log.info("[PASSWORD:SERVICE] 사용자 '{}'의 비밀번호가 변경되었습니다.", customer_id);
    }

    @Override
    public UserDetails loadUserByUsername(String id) throws UsernameNotFoundException {

        // ✅ 하드코딩 관리자 계정 (최소 구현)
        if ("admin".equals(id)) {
            return org.springframework.security.core.userdetails.User.builder()
                    .username("admin")
                    .password(passwordEncoder.encode("admin123"))
                    .roles("ADMIN")
                    .build();
        }

        CustomersEntity user = customersRepository.findById(id)
                .orElseThrow(() -> new UsernameNotFoundException("사용자를 찾을 수 없습니다 : " + id));

        withdrawalRepository.findById(user.getIdx())
                .filter(w -> Boolean.TRUE.equals(w.getIsDeleted()))
                .ifPresent(w -> {
                    log.warn("[LOGIN:SERVICE] 탈퇴한 계정 접근 시도: {}", id);
                    throw new UsernameNotFoundException("탈퇴 처리된 계정입니다: " + id);
                });

        return org.springframework.security.core.userdetails.User.builder()
                .username(user.getId())
                .password(user.getPassword())
                .roles("USER")
                .build();
    }
}
