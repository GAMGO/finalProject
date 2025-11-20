package org.iclass.finalproject.store.dto;

public class StoreCreateRequest {

    private String storeName;   // 노점 이름/간단 설명
    private Long foodTypeId;    // FOOD_INFO.IDX
    private String storeAddress;
    private Double lat;
    private Double lng;

    public StoreCreateRequest() {}

    public String getStoreName() { return storeName; }
    public void setStoreName(String storeName) { this.storeName = storeName; }

    public Long getFoodTypeId() { return foodTypeId; }
    public void setFoodTypeId(Long foodTypeId) { this.foodTypeId = foodTypeId; }

    public String getStoreAddress() { return storeAddress; }
    public void setStoreAddress(String storeAddress) { this.storeAddress = storeAddress; }

    public Double getLat() { return lat; }
    public void setLat(Double lat) { this.lat = lat; }

    public Double getLng() { return lng; }
    public void setLng(Double lng) { this.lng = lng; }
}
