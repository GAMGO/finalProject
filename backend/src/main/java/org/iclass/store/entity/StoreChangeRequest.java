package org.iclass.store.entity;

import jakarta.persistence.*;
import lombok.*;

@Entity
@Table(name = "store_change_request")
@NoArgsConstructor(access = AccessLevel.PROTECTED)
@AllArgsConstructor
@Builder
public class StoreChangeRequest {

    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }

    public Store getStore() { return store; }
    public void setStore(Store store) { this.store = store; }

    public org.iclass.store.enums.StoreChangeType getType() { return type; }
    public void setType(org.iclass.store.enums.StoreChangeType type) { this.type = type; }

    public org.iclass.store.enums.StoreChangeStatus getStatus() { return status; }
    public void setStatus(org.iclass.store.enums.StoreChangeStatus status) { this.status = status; }

    public Long getRequestedBy() { return requestedBy; }
    public void setRequestedBy(Long requestedBy) { this.requestedBy = requestedBy; }

    public java.time.LocalDateTime getRequestedAt() { return requestedAt; }
    public void setRequestedAt(java.time.LocalDateTime requestedAt) { this.requestedAt = requestedAt; }

    public Long getReviewedBy() { return reviewedBy; }
    public void setReviewedBy(Long reviewedBy) { this.reviewedBy = reviewedBy; }

    public java.time.LocalDateTime getReviewedAt() { return reviewedAt; }
    public void setReviewedAt(java.time.LocalDateTime reviewedAt) { this.reviewedAt = reviewedAt; }

    public String getRejectReason() { return rejectReason; }
    public void setRejectReason(String rejectReason) { this.rejectReason = rejectReason; }

    public String getNewStoreName() { return newStoreName; }
    public void setNewStoreName(String newStoreName) { this.newStoreName = newStoreName; }

    public String getNewOpenTime() { return newOpenTime; }
    public void setNewOpenTime(String newOpenTime) { this.newOpenTime = newOpenTime; }

    public String getNewCloseTime() { return newCloseTime; }
    public void setNewCloseTime(String newCloseTime) { this.newCloseTime = newCloseTime; }

    public String getNewStoreAddress() { return newStoreAddress; }
    public void setNewStoreAddress(String newStoreAddress) { this.newStoreAddress = newStoreAddress; }

    public Double getNewLat() { return newLat; }
    public void setNewLat(Double newLat) { this.newLat = newLat; }

    public Double getNewLng() { return newLng; }
    public void setNewLng(Double newLng) { this.newLng = newLng; }

    /* ========================= Fields ========================= */
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @ManyToOne(optional = false)
    @JoinColumn(name = "store_idx")
    private Store store;

    @Enumerated(EnumType.STRING)
    @Column(nullable = false, length = 20)
    private org.iclass.store.enums.StoreChangeType type;

    @Enumerated(EnumType.STRING)
    @Column(nullable = false, length = 20)
    private org.iclass.store.enums.StoreChangeStatus status;

    private Long requestedBy;

    @Column(nullable = false)
    private java.time.LocalDateTime requestedAt;

    private Long reviewedBy;
    private java.time.LocalDateTime reviewedAt;

    @Column(length = 1000)
    private String rejectReason;

    // 변경 예정 값들
    private String newStoreName;

    @Column(length = 5)
    private String newOpenTime;

    @Column(length = 5)
    private String newCloseTime;

    private String newStoreAddress;

    private Double newLat;
    private Double newLng;
}
