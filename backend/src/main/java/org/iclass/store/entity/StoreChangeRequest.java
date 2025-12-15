package org.iclass.store.entity;

import jakarta.persistence.*;
import lombok.Getter;
import lombok.Setter;
import org.iclass.store.enums.StoreChangeStatus;
import org.iclass.store.enums.StoreChangeType;

import java.time.LocalDateTime;

@Getter
@Setter
@Entity
@Table(name = "store_change_requests")
public class StoreChangeRequest {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name = "idx")
    private Long id; // ✅ getId() 로 맞춤 (네 코드가 getId() 쓰는 스타일)

    @Enumerated(EnumType.STRING)
    @Column(name = "type", nullable = false)
    private StoreChangeType type;

    @Enumerated(EnumType.STRING)
    @Column(name = "status", nullable = false)
    private StoreChangeStatus status;

    // CREATE는 null 가능, UPDATE/DELETE는 존재
    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "store_idx", foreignKey = @ForeignKey(name = "fk_scr_store"))
    private Store store;

    @Column(name = "requested_by", nullable = false)
    private Long requestedBy;

    @Column(name = "requested_at", nullable = false)
    private LocalDateTime requestedAt;

    @Column(name = "reviewed_by")
    private Long reviewedBy;

    @Column(name = "reviewed_at")
    private LocalDateTime reviewedAt;

    @Column(name = "reject_reason")
    private String rejectReason;

    // ===== 요청된 새 값들 =====
    @Column(name = "new_store_name")
    private String newStoreName;

    @Column(name = "new_store_address")
    private String newStoreAddress;

    @Column(name = "new_lat")
    private Double newLat;

    @Column(name = "new_lng")
    private Double newLng;

    @Column(name = "new_food_type_id")
    private Long newFoodTypeId;
}
