package org.iclass.store.entity;

import lombok.*;

import org.iclass.store.enums.StoreChangeStatus;
import org.iclass.store.enums.StoreChangeType;

import java.time.LocalDateTime;

@Getter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class StoreChangeRequestResponse {

    private Long id;
    private Long storeIdx;
    private StoreChangeType type;
    private StoreChangeStatus status;

    private Long requestedBy;
    private LocalDateTime requestedAt;

    private Long reviewedBy;
    private LocalDateTime reviewedAt;
    private String rejectReason;

    // === 변경 요청된 새로운 값들 ===
    private String newStoreName;
    private LocalDateTime newOpenTime;
    private LocalDateTime newCloseTime;
    private String newStoreAddress;
    private Long newFoodTypeId;
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
                .newFoodTypeId(r.getNewFoodTypeId())
                .newLat(r.getNewLat())
                .newLng(r.getNewLng())
                .build();
    }
}
