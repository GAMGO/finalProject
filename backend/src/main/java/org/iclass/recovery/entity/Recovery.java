package org.iclass.recovery.entity;

import org.iclass.customer.entity.CustomersEntity;
import jakarta.persistence.*;
import lombok.*;
import java.time.LocalDateTime;

@Getter 
@Setter 
@NoArgsConstructor 
@AllArgsConstructor 
@Builder
@Entity
@Table(name = "recovery",
       indexes = {
         @Index(name = "ix_rc_customer", columnList = "customer_idx")
       })
public class Recovery {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name = "idx")
    private Long idx;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "customer_idx", foreignKey = @ForeignKey(name = "fk_rc_customer"))
    private CustomersEntity customer;

    @Column(name = "email_verified_code", length = 32)
    private String emailVerifiedCode;

    @Column(name = "updated_at")
    private LocalDateTime updatedAt;
}
