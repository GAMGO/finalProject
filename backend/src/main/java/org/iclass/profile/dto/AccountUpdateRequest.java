// src/main/java/org/iclass/profile/dto/AccountUpdateRequest.java
package org.iclass.profile.dto;

import org.iclass.customer.entity.Gender;
import jakarta.validation.constraints.Size;
import lombok.Data;

@Data
public class AccountUpdateRequest {

    // 수정 가능한 필드만 둔다 (ID/Email은 수정 X)
    private String birth;         // "YYYY-MM-DD" 같은 문자열

    private Integer age;          // 프론트에서 계산해서 보내도 되고, 안 보내면 그대로 둠

    private Gender gender;        // M / F / null(비공개)

    @Size(max = 255)
    private String address;
}
