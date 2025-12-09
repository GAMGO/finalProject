package org.iclass.community;

import jakarta.persistence.*;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

import java.time.LocalDateTime;

@Entity
@Table(name = "community") 
@Getter
@Setter
@NoArgsConstructor
public class Post {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long idx;   // PK

    @Column(nullable = false)
    private String title;   // 제목

    @Lob
    private String body;    // 본문

    private String writer;          // 작성자
    private String category;        // 제보 / 후기 / 질문 등
    private String storeCategory;   // 음식 카테고리
    private String locationText;    // 위치
    private String imageUrl;        // 이미지 URL

    @Column(name = "created_at", updatable = false)
    private LocalDateTime createdAt;

    @Column(name = "updated_at")
    private LocalDateTime updatedAt;

    // 기본값 0 + null 방지
    @Column(name = "comment_count", nullable = false)
    private int commentCount = 0;   // 댓글 수

    @Column(name = "views", nullable = false)
    private int views = 0;          // 조회수

    @PrePersist
    public void onCreate() {
        this.createdAt = LocalDateTime.now();
        //  null 방지용: 혹시 값이 비어 있으면 0으로 초기화
        if (this.commentCount < 0) this.commentCount = 0;
        if (this.views < 0) this.views = 0;
    }

    @PreUpdate
    public void onUpdate() {
        this.updatedAt = LocalDateTime.now();
    }

    public void update(String title, String body, String category,
                       String storeCategory, String locationText, String imageUrl) {
        this.title = title;
        this.body = body;
        this.category = category;
        this.storeCategory = storeCategory;
        this.locationText = locationText;
        this.imageUrl = imageUrl;
    }
}
