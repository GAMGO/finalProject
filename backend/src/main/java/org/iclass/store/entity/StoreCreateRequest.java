package org.iclass.store.entity;

import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
public class StoreCreateRequest {
    private String storeName;
    private Long foodTypeId;
    private String storeAddress;
    private Double lat;
    private Double lng;
}
