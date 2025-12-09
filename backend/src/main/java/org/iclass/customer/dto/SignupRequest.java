package org.iclass.customer.dto;

import java.time.LocalDate;

import jakarta.validation.constraints.*;
import lombok.*;


@Getter @Setter @NoArgsConstructor @AllArgsConstructor @Builder
public class SignupRequest {
    @NotBlank
    private String id;

    @NotBlank
    private String password;

    @Email
    @NotBlank
    private String email;

    @Positive
    @Max(150)
    private Integer age;

    private LocalDate birth;

    private String address;
}
