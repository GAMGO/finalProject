package com.finalproject.backend.domain.chat;

import jakarta.persistence.*;
import lombok.*;

import java.time.LocalDateTime;

@Entity
@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class ChatMessage {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @ManyToOne(fetch = FetchType.LAZY)
    private ChatSession session;

    @Enumerated(EnumType.STRING)
    private SenderType sender;   // USER / AI

    @Lob
    private String content;      // 채팅 텍스트 내용

    // ===== 파일 / 이미지 첨부용 =====
    private String fileName;     // 원본 파일명
    private String fileUrl;      // /uploads/xxx 형태의 URL
    private String fileType;     // "FILE" / "IMAGE" 등

    private LocalDateTime createdAt;

    @PrePersist
    public void onCreate() {
        this.createdAt = LocalDateTime.now();
    }
}
