package org.iclass.customer.dto;

import java.time.LocalDate;

import org.iclass.customer.entity.Gender;
import jakarta.validation.constraints.*;
import lombok.*;


@Getter @Setter @NoArgsConstructor @AllArgsConstructor @Builder
public class SignupRequest {
    @NotBlank
    private String customer_id;

    @NotBlank
    @Size(min = 8, max = 72)
    private String password;

    @Email
    @NotBlank
    private String email;

    @Positive
    @Max(150)
    private Integer age;
    
    private Gender gender;

    private LocalDate birth;

    private String address;
}
