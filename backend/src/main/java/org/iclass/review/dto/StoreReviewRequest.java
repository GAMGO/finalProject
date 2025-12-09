// src/main/java/org/iclass/review/dto/StoreReviewRequest.java
package org.iclass.review.dto;

import jakarta.validation.constraints.*;

public class StoreReviewRequest {

    @NotNull @Min(1) @Max(5)
    private Integer rating;

    @Size(max = 10000)
    private String reviewText;

    private String aiTopics;

    // === getter / setter ===
    public Integer getRating() {
        return rating;
    }
    public void setRating(Integer rating) {
        this.rating = rating;
    }

    public String getReviewText() {
        return reviewText;
    }
    public void setReviewText(String reviewText) {
        this.reviewText = reviewText;
    }

    public String getAiTopics() {
        return aiTopics;
    }
    public void setAiTopics(String aiTopics) {
        this.aiTopics = aiTopics;
    }
}
