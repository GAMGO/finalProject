package com.finalproject.backend.dto;

import lombok.*;

@Getter @Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class ChatReplyDto {
    private ChatMessageDto userMessage;
    private ChatMessageDto assistantMessage;
}
