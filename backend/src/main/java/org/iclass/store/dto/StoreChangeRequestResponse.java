package org.iclass.store.dto;

import lombok.Getter;
import lombok.Setter;
import org.iclass.store.entity.StoreChangeRequest;
import org.iclass.store.enums.StoreChangeStatus;
import org.iclass.store.enums.StoreChangeType;

import java.time.LocalDateTime;

@Getter
@Setter
public class StoreChangeRequestResponse {

    private Long id;
    private StoreChangeType type;
    private StoreChangeStatus status;

    private Long storeIdx; // UPDATE/DELETE 대상 store (CREATE는 null)

    private String newStoreName;
    private String newStoreAddress;
    private Double newLat;
    private Double newLng;
    private Long newFoodTypeId;

    private Long requestedBy;
    private LocalDateTime requestedAt;

    private Long reviewedBy;
    private LocalDateTime reviewedAt;

    private String rejectReason;

    public static StoreChangeRequestResponse from(StoreChangeRequest e) {
        StoreChangeRequestResponse dto = new StoreChangeRequestResponse();
        dto.id = e.getId();
        dto.type = e.getType();
        dto.status = e.getStatus();

        dto.storeIdx = (e.getStore() == null ? null : e.getStore().getIdx());

        dto.newStoreName = e.getNewStoreName();
        dto.newStoreAddress = e.getNewStoreAddress();
        dto.newLat = e.getNewLat();
        dto.newLng = e.getNewLng();
        dto.newFoodTypeId = e.getNewFoodTypeId();

        dto.requestedBy = e.getRequestedBy();
        dto.requestedAt = e.getRequestedAt();
        dto.reviewedBy = e.getReviewedBy();
        dto.reviewedAt = e.getReviewedAt();
        dto.rejectReason = e.getRejectReason();
        return dto;
    }
}
