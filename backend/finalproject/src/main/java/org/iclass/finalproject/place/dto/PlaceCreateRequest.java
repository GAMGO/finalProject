package org.iclass.finalproject.place.dto;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import lombok.Data;
import org.iclass.finalproject.place.model.PlaceCategory;

@Data
public class PlaceCreateRequest {

    @NotNull
    private PlaceCategory category;

    private String name;

    @NotBlank
    private String address;

    @NotNull
    private Double latitude;

    @NotNull
    private Double longitude;

    private String shortDescription;
    private String mainMenu;
    private Integer priceLevel;
    private String tags;
}

/*
 * [파일 설명]
 * - 장소 등록 API 요청 바디 DTO.
 * - 유저가 발견한 포장마차/백반집을 등록할 때 필요한 정보를 정의.
 */
