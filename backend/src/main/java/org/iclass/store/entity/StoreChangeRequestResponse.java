package org.iclass.store.entity;

import lombok.*;
import org.iclass.store.enums.StoreChangeStatus;
import org.iclass.store.enums.StoreChangeType;

@Getter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class StoreChangeRequestResponse {

    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }

    public Long getStoreIdx() { return storeIdx; }
    public void setStoreIdx(Long storeIdx) { this.storeIdx = storeIdx; }

    public StoreChangeType getType() { return type; }
    public void setType(StoreChangeType type) { this.type = type; }

    public StoreChangeStatus getStatus() { return status; }
    public void setStatus(StoreChangeStatus status) { this.status = status; }

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
    private Long id;
    private Long storeIdx;
    private StoreChangeType type;
    private StoreChangeStatus status;

    private Long requestedBy;
    private java.time.LocalDateTime requestedAt;

    private Long reviewedBy;
    private java.time.LocalDateTime reviewedAt;
    private String rejectReason;

    private String newStoreName;
    private String newOpenTime;     
    private String newCloseTime;    
    private String newStoreAddress;

    private Double newLat;
    private Double newLng;

    public static StoreChangeRequestResponse from(StoreChangeRequest r) {
        return StoreChangeRequestResponse.builder()
                .id(r.getId())
                .storeIdx(r.getStore().getIdx())
                .type(r.getType())
                .status(r.getStatus())
                .requestedBy(r.getRequestedBy())
                .requestedAt(r.getRequestedAt())
                .reviewedBy(r.getReviewedBy())
                .reviewedAt(r.getReviewedAt())
                .rejectReason(r.getRejectReason())
                .newStoreName(r.getNewStoreName())
                .newOpenTime(r.getNewOpenTime())
                .newCloseTime(r.getNewCloseTime())
                .newStoreAddress(r.getNewStoreAddress())
                .newLat(r.getNewLat())
                .newLng(r.getNewLng())
                .build();
    }
}
