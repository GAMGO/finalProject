package org.iclass.customer.service;

import java.time.LocalDateTime;
import java.util.Date;
import java.util.List;
import java.util.Optional;

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

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;

@Slf4j
@RequiredArgsConstructor
@Service
public class CustomersService implements UserDetailsService {

        private final CustomersRepository customersRepository;
        private final PasswordEncoder passwordEncoder;
        private final EmailService emailService;

        @Transactional
        public CustomersEntity signup(SignupRequest req) {
                log.debug("[SIGNUP:SERVICE] existsById? id={}", req.getId()); // log 확인
                if (customersRepository.existsById(req.getId())) {
                        log.warn("[SIGNUP:SERVICE] duplicate id={}", req.getId());
                        // 중복 ID인 경우 기존 사용자 정보 반환 (이메일 인증만 재발송)
                        CustomersEntity existingUser = customersRepository.findById(req.getId()).orElse(null);
                        if (existingUser != null && !existingUser.getEmailVerified()) {
                                // 이메일 인증이 안된 경우 인증번호 재생성
                                String newCode = emailService.generateVerificationCode(); // UUID 대신 6자리 숫자 생성
                                LocalDateTime newExpires = LocalDateTime.now().plusMinutes(5); // 만료 시간 5분으로 설정
                                existingUser.setEmailVerificationToken(newCode); // 인증번호 저장
                                existingUser.setEmailVerificationExpires(newExpires);
                                customersRepository.save(existingUser);

                                // 이메일 재발송
                                emailService.sendVerificationEmail(req.getId(), newCode); // 6자리 인증번호를 발송
                                log.info("[SIGNUP:SERVICE] 이메일 인증번호 재발송: {}", req.getEmail());
                        }
                        return existingUser;
                }

                // 이메일 인증번호 생성
                String verificationCode = emailService.generateVerificationCode();
                LocalDateTime codeExpires = LocalDateTime.now().plusMinutes(5);

                CustomersEntity user = CustomersEntity.builder()
                                .id(req.getId())
                                .password(passwordEncoder.encode(req.getPassword())) // 비밀번호 암호화
                                .age(req.getAge())
                                .address(req.getAddress())
                                .birth(req.getBirth().toString())
                                .email(req.getEmail())
                                .emailVerified(false) // 이메일 인증 대기 상태
                                .emailVerificationToken(verificationCode) // 인증번호
                                .emailVerificationExpires(codeExpires) // 인증번호 만료 시간
                                .build();

                CustomersEntity savedUser = customersRepository.save(user);

                // 이메일 인증번호 발송
                boolean emailSent = emailService.sendVerificationEmail(req.getEmail(), verificationCode);
                if (emailSent) {
                        log.info("[SIGNUP:SERVICE] 이메일 인증번호 발송 성공: {}", req.getEmail());
                } else {
                        log.warn("[SIGNUP:SERVICE] 이메일 인증번호 발송 실패: {}", req.getEmail());
                }

                return savedUser;
        }

        // 복구/프로필 등에서 공용으로 쓰는 비밀번호 변경 유틸
        @Transactional
        public void updatePassword(String customer_id, String newPassword) {
                CustomersEntity user = customersRepository.findById(customer_id)
                                .orElseThrow(() -> new UsernameNotFoundException("사용자를 찾을 수 없습니다 : " + customer_id));
                user.setPassword(passwordEncoder.encode(newPassword));
                customersRepository.save(user);
        }

        @Transactional
        public void deleteCustomer(String id) {
                // 1. 사용자 조회
                CustomersEntity user = customersRepository.findById(id)
                                .orElseThrow(() -> new UsernameNotFoundException("사용자를 찾을 수 없습니다: " + id));

                // 2. 소프트 삭제 처리: 필드값 변경 및 저장
                user.setIsDeleted(true);
                user.setRefreshToken(null); // 보안을 위해 리프레시 토큰도 무효화
                customersRepository.save(user);

                log.info("[WITHDRAWAL:SERVICE] 사용자 소프트 삭제 완료: {}", id);
        }

        @Override
        public UserDetails loadUserByUsername(String id) throws UsernameNotFoundException {
                CustomersEntity user = customersRepository.findById(id)
                                .orElseThrow(() -> new UsernameNotFoundException("사용자를 찾을 수 없습니다 : " + id));
                // 3. 탈퇴 계정 로그인 차단 로직 추가
                if (Boolean.TRUE.equals(user.getIsDeleted())) {
                        log.warn("[LOGIN:SERVICE] 탈퇴한 계정 접근 시도: {}", id);
                        throw new UsernameNotFoundException("탈퇴 처리된 계정입니다: " + id);
                }
                return org.springframework.security.core.userdetails.User.builder()
                                .username(user.getId())
                                .password(user.getPassword())
                                .roles("USER")
                                .build();
        }
}