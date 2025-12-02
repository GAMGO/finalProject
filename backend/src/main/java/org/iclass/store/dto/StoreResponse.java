package org.iclass.store.dto;

import org.iclass.store.entity.Store;

public class StoreResponse {

    private Long idx;
    private String storeName;
    private String address;
    private Double latitude;
    private Double longitude;

    public StoreResponse() {}

    public static StoreResponse from(Store store) {
        StoreResponse dto = new StoreResponse();
        dto.idx = store.getIdx();
        dto.storeName = store.getStoreName();
        dto.address = store.getStoreAddress();
        dto.latitude = store.getLat();
        dto.longitude = store.getLng();
        return dto;
    }

    // ===== Getter / Setter =====

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
}
