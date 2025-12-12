package org.iclass.store.entity;

import jakarta.persistence.*;
import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
@Entity
@Table(name = "STORE")
public class Store {

    public Long getIdx() { return idx; }
    public void setIdx(Long idx) { this.idx = idx; }

    public String getStoreName() { return storeName; }
    public void setStoreName(String storeName) { this.storeName = storeName; }

    public String getOpenTime() { return openTime; }
    public void setOpenTime(String openTime) { this.openTime = openTime; }

    public String getCloseTime() { return closeTime; }
    public void setCloseTime(String closeTime) { this.closeTime = closeTime; }

    public String getStoreAddress() { return storeAddress; }
    public void setStoreAddress(String storeAddress) { this.storeAddress = storeAddress; }

    public Double getLat() { return lat; }
    public void setLat(Double lat) { this.lat = lat; }

    public Double getLng() { return lng; }
    public void setLng(Double lng) { this.lng = lng; }

    // ✅ 추가
    public Long getFoodTypeId() { return foodTypeId; }
    public void setFoodTypeId(Long foodTypeId) { this.foodTypeId = foodTypeId; }

    /* ========================= Fields ========================= */
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name = "IDX")
    private Long idx;

    @Column(name = "STORE_NAME", length = 255, nullable = false)
    private String storeName;

    @Column(name = "OPENTIME", length = 5)
    private String openTime;

    @Column(name = "CLOSETIME", length = 5)
    private String closeTime;

    @Column(name = "STORE_ADDRESS", length = 255, nullable = false)
    private String storeAddress;

    @Column(name = "LAT", nullable = false)
    private Double lat;

    @Column(name = "LNG", nullable = false)
    private Double lng;

    // ✅ 핵심: 카테고리 id 컬럼
    @Column(name = "FOOD_TYPE_ID")
    private Long foodTypeId;
}
