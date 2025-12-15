package org.iclass.store.dto;

import lombok.Getter;
import lombok.Setter;
import org.iclass.store.entity.Store;

@Getter
@Setter
public class StoreResponse {

    private Long idx;
    private String storeName;

    // 프론트에서 address/latitude/longitude로 쓰는 경우가 많아서 이렇게 맞춤
    private String address;
    private Double latitude;
    private Double longitude;

    private Long foodTypeId;

    public static StoreResponse fromEntity(Store store) {
        StoreResponse dto = new StoreResponse();
        dto.idx = store.getIdx();
        dto.storeName = store.getStoreName();
        dto.address = store.getStoreAddress();
        dto.latitude = store.getLat();
        dto.longitude = store.getLng();
        dto.foodTypeId = store.getFoodTypeId();
        return dto;
    }

    // 혹시 기존 코드가 from()을 쓰면 이것도 열어둠
    public static StoreResponse from(Store store) {
        return fromEntity(store);
    }
}
