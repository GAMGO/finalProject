package org.iclass.finalproject.review.dto;

import lombok.Builder;
import lombok.Data;

import java.time.LocalDateTime;

@Data
@Builder
public class ReviewResponse {
    private Long id;
    private Long placeId;
    private Long userId;
    private String username;
    private boolean owner;
    private int rating;
    private String content;
    private int likeCount;
    private LocalDateTime createdAt;
}

/*
 * [파일 설명]
 * - 리뷰 목록/상세에서 프론트에 내려줄 응답 DTO.
 * - "사장 리뷰인지", "좋아요 수", "작성자 닉네임" 등의 표시용 데이터를 포함.
 */
