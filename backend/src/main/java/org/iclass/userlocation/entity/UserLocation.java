package org.iclass.userlocation.entity;

import org.iclass.customer.entity.CustomersEntity;
import jakarta.persistence.*;
import lombok.*;
import java.math.BigDecimal;
import java.time.LocalDateTime;

@Getter
@Setter 
@NoArgsConstructor 
@AllArgsConstructor 
@Builder
@Entity
@Table(name = "user_location",
       indexes = {
         @Index(name = "ix_ul_customer", columnList = "customer_idx")
       })
public class UserLocation {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name = "idx")
    private Long idx;

    @Column(name = "record_time")
    private LocalDateTime recordTime;

    @Column(name = "located_at", length = 255)
    private String locatedAt;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "customer_idx", foreignKey = @ForeignKey(name = "fk_ul_customer"))
    private CustomersEntity customer;

    @Column(name = "lat", precision = 10, scale = 6)
    private BigDecimal lat;

    @Column(name = "lng", precision = 10, scale = 6)
    private BigDecimal lng;
}
