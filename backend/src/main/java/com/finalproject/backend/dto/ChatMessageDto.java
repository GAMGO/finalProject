package com.finalproject.backend.dto;

import lombok.*;

@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class ChatMessageDto {

    private Long id;
    private String sender;
    private String content;

    private String fileName;
    private String fileUrl;
    private String fileType;
}
