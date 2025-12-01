package org.iclass.store;

import jakarta.persistence.*;
import lombok.Getter;
import lombok.Setter;

import java.time.LocalDateTime;
@Getter
@Setter
@Entity
@Table(name = "STORE")
public class Store {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name = "IDX")
    private Long idx;

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
}
