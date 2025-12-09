// src/main/java/org/iclass/gemini/dto/ModerationResult.java
package org.iclass.gemini.dto;

import lombok.Data;

@Data
public class ModerationResult {
    // "ALLOW", "BLOCK", "REVIEW" 중 하나
    private String decision;
    private String reason;

    public boolean isBlocked() {
        return "BLOCK".equalsIgnoreCase(decision);
    }

    public boolean needManualReview() {
        return "REVIEW".equalsIgnoreCase(decision);
    }
}
