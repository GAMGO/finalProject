package org.iclass.comment.dto;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Size;
import lombok.Getter;
import lombok.Setter;

@Getter @Setter
public class CommentRequest {

    @Size(max = 100)
    private String author;     // 선택

    @NotBlank
    private String content;    // 필수
}
