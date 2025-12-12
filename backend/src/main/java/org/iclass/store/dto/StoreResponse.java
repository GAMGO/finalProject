package org.iclass.store.dto;

import org.iclass.store.FoodCategory;
import org.iclass.store.entity.Store;

public class StoreResponse {

    private Long idx;
    private String storeName;
    private String address;
    private Double latitude;
    private Double longitude;

    // ✅ 추가
    private Long foodTypeId;
    private String foodTypeLabel;

    public StoreResponse() {}

    public static StoreResponse from(Store store) {
        StoreResponse dto = new StoreResponse();
        dto.idx = store.getIdx();
        dto.storeName = store.getStoreName();
        dto.address = store.getStoreAddress();
        dto.latitude = store.getLat();
        dto.longitude = store.getLng();

        // ✅ 이제 가능
        dto.foodTypeId = store.getFoodTypeId();
        dto.foodTypeLabel = FoodCategory.labelOf(dto.foodTypeId);

        return dto;
    }

    public Long getIdx() { return idx; }
    public void setIdx(Long idx) { this.idx = idx; }

    public String getStoreName() { return storeName; }
    public void setStoreName(String storeName) { this.storeName = storeName; }

    public String getAddress() { return address; }
    public void setAddress(String address) { this.address = address; }

    public Double getLatitude() { return latitude; }
    public void setLatitude(Double latitude) { this.latitude = latitude; }

    public Double getLongitude() { return longitude; }
    public void setLongitude(Double longitude) { this.longitude = longitude; }

    public Long getFoodTypeId() { return foodTypeId; }
    public void setFoodTypeId(Long foodTypeId) { this.foodTypeId = foodTypeId; }

    public String getFoodTypeLabel() { return foodTypeLabel; }
    public void setFoodTypeLabel(String foodTypeLabel) { this.foodTypeLabel = foodTypeLabel; }
}
