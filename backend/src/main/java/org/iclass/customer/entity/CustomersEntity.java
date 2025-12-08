package org.iclass.customer.entity;

import jakarta.persistence.*;
import jakarta.validation.constraints.Email;
import lombok.*;
import java.time.LocalDateTime;
import com.fasterxml.jackson.annotation.JsonFormat;

@Getter
@Setter
@Builder
@NoArgsConstructor
@AllArgsConstructor
@Entity
@Table(name = "customers")
public class CustomersEntity {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name = "idx")
    private Long idx;

    @Column(name = "customer_id", length = 100, nullable = false, updatable = false, unique = true)
    private String id;

    @Column(name = "password_hash", length = 255, nullable = false)
    private String password;

    @Email
    @Column(name = "email")
    private String email;

    @Column(name = "age")
    private Integer age;

    @Column(name = "birth")
    private String birth;

    @Column(name = "address")
    private String address;

    @Column(name = "email_verified")
    private Boolean emailVerified; // 이메일 인증

    @Column(name = "email_verification_token")
    private String emailVerificationToken; // 이메일 인증 토큰

    @Column(name = "email_verification_expires")
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime emailVerificationExpires;

    @Column(name = "refresh_token", length = 1024)
    private String refreshToken;
    
    @Column(name = "is_deleted", nullable = false)
    private Boolean isDeleted = false;
}