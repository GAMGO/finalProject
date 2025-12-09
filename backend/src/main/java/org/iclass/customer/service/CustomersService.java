package org.iclass.customer.service;

import java.time.LocalDateTime;
import java.util.Date;
import java.util.List;
import java.util.Optional;

import org.iclass.emailVerification.service.EmailRecoveryService;
import org.iclass.emailVerification.service.EmailService;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.security.core.userdetails.UserDetailsService;
import org.springframework.security.core.userdetails.UsernameNotFoundException;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import org.iclass.customer.dto.SignupRequest;
import org.iclass.customer.entity.CustomersEntity;
import org.iclass.customer.repository.CustomersRepository;
import org.iclass.deleteAccount.entity.WithdrawalEntity;
import org.iclass.deleteAccount.repository.WithdrawalRepository;

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
                // 1. ID 중복 체크 (선 인증 후 가입 흐름에서는 중복 ID 체크만 하고, 인증 재발송 로직은 제거)
                if (customersRepository.existsById(req.getId())) {
                        log.warn("[SIGNUP:SERVICE] duplicate id={}", req.getId());
                        // 기존 사용자가 인증된 상태라면 (혹은 프론트에서 재인증을 유도했다면)
                        // 여기서는 에러를 발생시켜야 프론트가 '이미 가입된 사용자'임을 인지하고 로그인으로 유도할 수 있습니다.
                        throw new IllegalStateException("이미 가입된 사용자 ID입니다.");
                }

                // 2. 새로운 사용자 생성 (인증 완료를 전제로 생성)
                CustomersEntity user = CustomersEntity.builder()
                                .id(req.getId())
                                .password(passwordEncoder.encode(req.getPassword())) // 비밀번호 암호화
                                .age(req.getAge())
                                .address(req.getAddress())
                                .birth(req.getBirth().toString())
                                .email(req.getEmail())
                                // ✅ [수정] 이메일 인증 성공 후 API가 호출되었으므로, DB에 '인증 완료(true)'로 저장합니다.
                                .emailVerified(true)
                                // ✅ [수정] 인증 코드는 EmailService 캐시에서 이미 검증되었으므로 DB 필드는 null 유지
                                .emailVerificationToken(null)
                                .emailVerificationExpires(null)
                                .build();

                CustomersEntity savedUser = customersRepository.save(user);
                log.info("[SIGNUP:SERVICE] 새로운 사용자 가입 및 인증 완료: {}", req.getId());
                return savedUser;
        }

        @Transactional
        public void deleteCustomer(String id) {
                // 1. 사용자 조회
                CustomersEntity user = customersRepository.findById(id)
                                .orElseThrow(() -> new UsernameNotFoundException("사용자를 찾을 수 없습니다: " + id));

                // 2. [수정] 탈퇴 테이블 정보 업데이트
                WithdrawalEntity withdrawal = withdrawalRepository.findById(user.getIdx())
                                .orElse(WithdrawalEntity.builder().customerIdx(user.getIdx()).build());

                String recoveryCode = emailService.generateVerificationCode(); // 6자리 복구 코드 생성

                withdrawal.setIsDeleted(true);
                withdrawal.setRecoveryToken(recoveryCode);
                withdrawal.setDeletedAt(LocalDateTime.now());

                withdrawalRepository.save(withdrawal);

                // 3. 보안 처리: 리프레시 토큰 무효화
                user.setRefreshToken(null);
                customersRepository.save(user);

                // 4. 탈퇴 통보 및 복구 메세지 전송
                emailRecoveryService.sendWithdrawalNotification(user.getEmail(), recoveryCode);

                log.info("[WITHDRAWAL:SERVICE] 사용자 소프트 삭제 처리 및 복구 메일 발송: {}", id);
        }

        // 본인 확인 후 계정 복구
        @Transactional
        public void restoreCustomer(String recoveryToken) {
                // 1. 토큰으로 탈퇴 정보 조회
                WithdrawalEntity withdrawal = withdrawalRepository.findByRecoveryToken(recoveryToken)
                                .orElseThrow(() -> new IllegalArgumentException("유효하지 않거나 만료된 복구 토큰입니다."));

                // 2. 상태 정상화
                withdrawal.setIsDeleted(false);
                withdrawal.setRecoveryToken(null); // 토큰 사용 후 삭제
                withdrawalRepository.save(withdrawal);

                log.info("[RESTORE:SERVICE] 계정 복구 성공 (고객 인덱스: {})", withdrawal.getCustomerIdx());
        }

        @Transactional
        public void updatePassword(String customer_id, String newPassword) {
                // 1. 사용자 조회
                CustomersEntity user = customersRepository.findById(customer_id)
                                .orElseThrow(() -> new UsernameNotFoundException("사용자를 찾을 수 없습니다 : " + customer_id));

                // 2. 새로운 비밀번호 암호화 및 영속화
                user.setPassword(passwordEncoder.encode(newPassword));
                customersRepository.save(user);

                log.info("[PASSWORD:SERVICE] 사용자 '{}'의 비밀번호가 변경되었습니다.", customer_id);
        }

        @Override
        public UserDetails loadUserByUsername(String id) throws UsernameNotFoundException {
                // 1. 사용자 조회
                CustomersEntity user = customersRepository.findById(id)
                                .orElseThrow(() -> new UsernameNotFoundException("사용자를 찾을 수 없습니다 : " + id));

                // 2. [수정] 별도 탈퇴 테이블에서 상태 확인
                withdrawalRepository.findById(user.getIdx())
                                .filter(WithdrawalEntity::getIsDeleted)
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