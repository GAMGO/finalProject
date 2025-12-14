package org.iclass.store.dto;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Size;

import java.time.LocalDateTime;

public class StoreCreateRequest {

    @NotBlank(message = "노점 이름(storeName)은 필수입니다.")
    @Size(max = 255, message = "노점 이름은 최대 255자까지 가능합니다.")
    private String storeName;

    @NotNull(message = "카테고리(foodTypeId)를 선택해주세요.")
    private Long foodTypeId;

    @NotBlank(message = "주소(storeAddress)는 필수입니다.")
    @Size(max = 255, message = "주소는 최대 255자까지 가능합니다.")
    private String storeAddress;

    @NotNull(message = "위도(lat)는 필수값입니다.")
    private Double lat;

    @NotNull(message = "경도(lng)는 필수값입니다.")
    private Double lng;

    // 선택값
    private LocalDateTime openTime;
    private LocalDateTime closeTime;

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

    public LocalDateTime getOpenTime() { return openTime; }
    public void setOpenTime(LocalDateTime openTime) { this.openTime = openTime; }

    public LocalDateTime getCloseTime() { return closeTime; }
    public void setCloseTime(LocalDateTime closeTime) { this.closeTime = closeTime; }
}
