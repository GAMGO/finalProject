package org.iclass.finalproject.review.dto;

import jakarta.validation.constraints.Max;
import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.NotBlank;
import lombok.Data;

@Data
public class ReviewCreateRequest {

    @Min(1) @Max(5)
    private int rating;

    @NotBlank
    private String content;
}

/*
 * [파일 설명]
 * - 리뷰 작성 API 요청 바디 DTO.
 * - 별점(1~5)과 리뷰 내용을 전달하는 역할.
 */
