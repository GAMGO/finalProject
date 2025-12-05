// src/main/java/org/iclass/profile/dto/AccountInfoDto.java
package org.iclass.profile.dto;

import lombok.Builder;
import lombok.Data;

@Data
@Builder
public class AccountInfoDto {
    private String id;
    private String email;
    private String birth;       // entity가 String이라 String으로 둠
    private Integer age;
    private String address;
    private Boolean emailVerified;
}
