package org.iclass.review.dto;

import jakarta.validation.constraints.*;

public class StoreReviewRequest {
    @NotNull @Min(1) @Max(5)
    public Integer rating;

    @Size(max = 10000)
    public String reviewText;

    public String aiTopics;
}
