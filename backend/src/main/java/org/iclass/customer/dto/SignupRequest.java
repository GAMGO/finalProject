package org.iclass.customer.dto;

import java.time.LocalDate;

import jakarta.validation.constraints.*;
import lombok.*;


@Getter @Setter @NoArgsConstructor @AllArgsConstructor @Builder
public class SignupRequest {
    @NotBlank
    private String id;

    @NotBlank
    @Size(min = 8, max = 72)
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
