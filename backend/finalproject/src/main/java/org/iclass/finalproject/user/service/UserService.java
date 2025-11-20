package org.iclass.finalproject.user.service;

import lombok.RequiredArgsConstructor;
import org.iclass.finalproject.user.dto.*;
import org.iclass.finalproject.user.model.User;
import org.iclass.finalproject.user.model.UserSecurityQuestion;
import org.iclass.finalproject.user.repository.UserRepository;
import org.iclass.finalproject.user.repository.UserSecurityQuestionRepository;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDate;
import java.util.List;

@Service
@RequiredArgsConstructor
public class UserService {

    private final UserRepository userRepository;
    private final UserSecurityQuestionRepository userSecurityQuestionRepository;
    private final PasswordEncoder passwordEncoder;

    // ========== 회원가입 / 프로필 ==========

    public User signup(SignupRequest req) {
        User u = new User();
        u.setUsername(req.getUsername());
        u.setEmail(req.getEmail());
        u.setPasswordHash(passwordEncoder.encode(req.getPassword()));
        return userRepository.save(u);
    }

    public User getById(Long id) {
        return userRepository.findById(id)
                .orElseThrow(() -> new IllegalArgumentException("user not found"));
    }

    public UserProfileResponse getProfile(Long id) {
        User u = getById(id);
        return UserProfileResponse.builder()
                .id(u.getId())
                .username(u.getUsername())
                .email(u.getEmail())
                .birthday(u.getBirthday())
                .gender(u.getGender())
                .address(u.getAddress())
                .bio(u.getBio())
                .build();
    }

    public void updateProfile(Long id, String username, String address,
                              String bio, LocalDate birthday, String gender) {
        User u = getById(id);
        if (username != null) u.setUsername(username);
        if (address != null) u.setAddress(address);
        if (bio != null) u.setBio(bio);
        if (birthday != null) u.setBirthday(birthday);
        if (gender != null) u.setGender(gender);
        userRepository.save(u);
    }

    // ========== 로그인 검증 ==========

    public User login(LoginRequest req) {
        User u = userRepository.findByEmail(req.getEmail())
                .orElseThrow(() -> new IllegalArgumentException("이메일 또는 비밀번호가 올바르지 않습니다."));

        if (!passwordEncoder.matches(req.getPassword(), u.getPasswordHash())) {
            throw new IllegalArgumentException("이메일 또는 비밀번호가 올바르지 않습니다.");
        }

        return u;
    }

    // ========== 비밀번호 변경 / 보안 질문 ==========

    @Transactional
    public void changePassword(Long userId, ChangePasswordRequest req) {
        User u = userRepository.findById(userId)
                .orElseThrow(() -> new IllegalArgumentException("user not found"));

        if (!passwordEncoder.matches(req.getCurrentPassword(), u.getPasswordHash())) {
            throw new IllegalArgumentException("현재 비밀번호가 일치하지 않습니다.");
        }

        u.setPasswordHash(passwordEncoder.encode(req.getNewPassword()));
        userRepository.save(u);
    }

    @Transactional
    public void setSecurityQuestions(Long userId, SetSecurityQuestionsRequest req) {
        User u = userRepository.findById(userId)
                .orElseThrow(() -> new IllegalArgumentException("user not found"));

        List<UserSecurityQuestion> existing = userSecurityQuestionRepository.findByUser(u);
        userSecurityQuestionRepository.deleteAll(existing);

        for (SecurityQuestionDto dto : req.getQuestions()) {
            UserSecurityQuestion q = new UserSecurityQuestion();
            q.setUser(u);
            q.setQuestion(dto.getQuestion());
            q.setAnswerHash(passwordEncoder.encode(dto.getAnswer().trim()));
            userSecurityQuestionRepository.save(q);
        }
    }

    @Transactional
    public void resetPasswordByQuestion(ResetPasswordByQuestionRequest req) {
        User u = userRepository.findByEmail(req.getEmail())
                .orElseThrow(() -> new IllegalArgumentException("해당 이메일의 사용자가 없습니다."));

        UserSecurityQuestion q = userSecurityQuestionRepository
                .findByUserAndQuestion(u, req.getQuestion())
                .orElseThrow(() -> new IllegalArgumentException("해당 질문이 설정되어 있지 않습니다."));

        if (!passwordEncoder.matches(req.getAnswer().trim(), q.getAnswerHash())) {
            throw new IllegalArgumentException("보안 질문 답변이 일치하지 않습니다.");
        }

        u.setPasswordHash(passwordEncoder.encode(req.getNewPassword()));
        userRepository.save(u);
    }
}

/*
 * [파일 설명]
 * - 유저 관련 비즈니스 로직 담당 서비스.
 * - 회원가입, 프로필 조회/수정, 로그인 검증,
 *   비밀번호 변경, 보안 질문 설정/검증, 비밀번호 재설정까지 포함.
 * - AuthController, MeController에서 이 서비스만 호출해서 유저 로직 처리.
 */
