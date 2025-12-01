package org.iclass.store.entity;

import jakarta.persistence.*;
import lombok.*;

import java.time.LocalDateTime;

import org.iclass.store.enums.StoreChangeStatus;
import org.iclass.store.enums.StoreChangeType;

@Entity
@Table(name = "store_change_request")
@Getter
@Setter
@NoArgsConstructor(access = AccessLevel.PROTECTED)
@AllArgsConstructor
@Builder
public class StoreChangeRequest {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    // 어떤 점포에 대한 요청인지
    @ManyToOne(optional = false)
    @JoinColumn(name = "store_idx")
    private Store store;

    @Enumerated(EnumType.STRING)
    @Column(nullable = false, length = 20)
    private StoreChangeType type;      // UPDATE or DELETE

    @Enumerated(EnumType.STRING)
    @Column(nullable = false, length = 20)
    private StoreChangeStatus status;  // PENDING, APPROVED, REJECTED

    // 요청자 (사용자 ID)
    private Long requestedBy;

    @Column(nullable = false)
    private LocalDateTime requestedAt;

    // 관리자 승인/거절 정보
    private Long reviewedBy;
    private LocalDateTime reviewedAt;

    @Column(length = 1000)
    private String rejectReason;

    // === UPDATE 요청 시 변경하고 싶은 값들 (Store 엔티티와 1:1 매칭) ===

    // STORE.STORE_NAME
    private String newStoreName;

    // STORE.OPENTIME / CLOSETIME
    private LocalDateTime newOpenTime;
    private LocalDateTime newCloseTime;

    // STORE.STORE_ADDRESS
    private String newStoreAddress;

    // STORE.FOOD_TYPE
    private Long newFoodTypeId;

    // STORE.LAT / LNG
    private Double newLat;
    private Double newLng;
}
