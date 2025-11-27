package org.iclass.finalproject.customer.dto;

import org.iclass.finalproject.customer.entity.CustomersEntity;
import org.iclass.finalproject.customer.entity.Gender;
import lombok.*;

@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class SignupResponse {
    private String id;
    private String email;
    private Integer age;
    private Gender gender;
    private String birth;
    private String address;
    private String accessToken; // JTW
    private String tokenType; // "Bearer"
    private long expiresIn; // 만료 시간(초) <- 추가해두면 프론트가 토큰 관리 편리함.

    private Boolean emailVerified; // 이메일 인증 상태 반환


    // AuthController 에서 엔티티를 그대로 반환하지 않고 DTO로 변환하기 위해 추가
    public static SignupResponse fromEntity(CustomersEntity entity) {
       return SignupResponse.builder()
                .id(entity.getId())
                .email(entity.getEmail())
                .age(entity.getAge())
                .gender(entity.getGender())
                .birth(entity.getBirth().toString())
                .address(entity.getAddress())
                .build();
    }


}
