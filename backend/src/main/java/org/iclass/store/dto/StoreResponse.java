package org.iclass.store.dto;

import org.iclass.store.FoodCategory;
import org.iclass.store.entity.Store;

public class StoreResponse {

    private Long idx;
    private String storeName;
    private Long foodTypeId;
    private String category;        // "통닭" 이런 한글 이름
    private String address;
    private Double latitude;
    private Double longitude;

    public StoreResponse() {}

    public static StoreResponse from(Store store) {
        StoreResponse dto = new StoreResponse();
        dto.idx = store.getIdx();
        dto.storeName = store.getStoreName();
        dto.foodTypeId = store.getFoodTypeId();
        dto.category = FoodCategory.labelOf(store.getFoodTypeId());
        dto.address = store.getStoreAddress();
        dto.latitude = store.getLat();
        dto.longitude = store.getLng();
        return dto;
    }

    // ===== Getter / Setter =====

    public Long getIdx() { return idx; }
    public void setId(Long idx) { this.idx = idx; }

    public String getStoreName() { return storeName; }
    public void setStoreName(String storeName) { this.storeName = storeName; }

    public Long getFoodTypeId() { return foodTypeId; }
    public void setFoodTypeId(Long foodTypeId) { this.foodTypeId = foodTypeId; }

    public String getCategory() { return category; }
    public void setCategory(String category) { this.category = category; }

    public String getAddress() { return address; }
    public void setAddress(String address) { this.address = address; }

    public Double getLatitude() { return latitude; }
    public void setLatitude(Double latitude) { this.latitude = latitude; }

    public Double getLongitude() { return longitude; }
    public void setLongitude(Double longitude) { this.longitude = longitude; }
}
