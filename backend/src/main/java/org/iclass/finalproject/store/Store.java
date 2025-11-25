package org.iclass.finalproject.store;

import jakarta.persistence.*;
import java.time.LocalDateTime;

@Entity
@Table(name = "STORE")
public class Store {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name = "IDX")
    private Long id;

    @Column(name = "STORE_NAME", length = 255, nullable = false)
    private String storeName;      // 노점 이름 + 한 줄 설명 느낌

    @Column(name = "OPENTIME")
    private LocalDateTime openTime;   // 아직 안 쓸 거면 null 허용

    @Column(name = "CLOSETIME")
    private LocalDateTime closeTime;

    @Column(name = "STORE_ADDRESS", length = 255, nullable = false)
    private String storeAddress;

    // FK: FOOD_INFO.IDX
    @Column(name = "FOOD_TYPE", nullable = false)
    private Long foodTypeId;

    @Column(name = "LAT", nullable = false)
    private Double lat;

    @Column(name = "LNG", nullable = false)
    private Double lng;

    public Store() {}

    // ===== Getter / Setter =====

    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }

    public String getStoreName() { return storeName; }
    public void setStoreName(String storeName) { this.storeName = storeName; }

    public LocalDateTime getOpenTime() { return openTime; }
    public void setOpenTime(LocalDateTime openTime) { this.openTime = openTime; }

    public LocalDateTime getCloseTime() { return closeTime; }
    public void setCloseTime(LocalDateTime closeTime) { this.closeTime = closeTime; }

    public String getStoreAddress() { return storeAddress; }
    public void setStoreAddress(String storeAddress) { this.storeAddress = storeAddress; }

    public Long getFoodTypeId() { return foodTypeId; }
    public void setFoodTypeId(Long foodTypeId) { this.foodTypeId = foodTypeId; }

    public Double getLat() { return lat; }
    public void setLat(Double lat) { this.lat = lat; }

    public Double getLng() { return lng; }
    public void setLng(Double lng) { this.lng = lng; }
}
