package org.iclass.finalproject.place.dto;

import lombok.Builder;
import lombok.Data;
import org.iclass.finalproject.place.model.PlaceCategory;
import org.iclass.finalproject.place.model.PlaceStatus;

@Data
@Builder
public class PlaceResponse {
    private Long id;
    private PlaceStatus status;
    private String name;
    private PlaceCategory category;
    private String address;
    private Double latitude;
    private Double longitude;
    private String shortDescription;
    private String mainMenu;
    private Integer priceLevel;
    private String tags;
    private Integer likeCount;
    private Double avgRating;
    private Integer reviewCount;
    private boolean favorite;
}

/*
 * [파일 설명]
 * - 장소 카드/상세 화면에서 사용할 응답 DTO.
 * - "현재 로그인한 유저가 즐겨찾기했는지 여부(favorite)"까지 포함해서 내려줌.
 */
