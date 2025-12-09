package org.iclass.deleteAccount.entity;

import jakarta.persistence.*;
import lombok.*;
import java.time.LocalDateTime;

@Getter
@Setter
@Builder
@NoArgsConstructor
@AllArgsConstructor
@Entity
@Table(name = "customer_withdrawals")
public class WithdrawalEntity {

    @Id
    private Long customerIdx; // CustomersEntity의 idx와 1:1 매핑

    @Column(nullable = false)
    private Boolean isDeleted = false;

    @Column(name = "recovery_token")
    private String recoveryToken;

    @Column(name = "deleted_at")
    private LocalDateTime deletedAt; // 통계용 탈퇴 일시 추가 가능
}