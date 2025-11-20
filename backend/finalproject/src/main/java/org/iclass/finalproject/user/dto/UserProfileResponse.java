package org.iclass.finalproject.user.dto;

import lombok.Builder;
import lombok.Data;

import java.time.LocalDate;

@Data
@Builder
public class UserProfileResponse {
    private Long id;
    private String username;
    private String email;
    private LocalDate birthday;
    private String gender;
    private String address;
    private String bio;
}

/*
 * [파일 설명]
 * - "내 프로필 보기" API 응답 DTO.
 * - 엔티티 전체를 그대로 노출하지 않고 필요한 정보만 전달하기 위한 뷰 모델 역할.
 */
