package org.iclass.finalproject.review.model;

import jakarta.persistence.*;
import lombok.Getter;
import lombok.Setter;
import org.iclass.finalproject.place.model.Place;
import org.iclass.finalproject.user.model.User;

import java.time.LocalDateTime;

@Entity
@Table(name = "reviews")
@Getter @Setter
public class Review {

    @Id @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @ManyToOne(fetch = FetchType.LAZY) @JoinColumn(name = "place_id", nullable = false)
    private Place place;

    @ManyToOne(fetch = FetchType.LAZY) @JoinColumn(name = "user_id", nullable = false)
    private User user;

    @Column(nullable = false)
    private int rating; // 1~5

    @Column(nullable = false, columnDefinition = "TEXT")
    private String content;

    @Column(name = "is_owner", nullable = false)
    private boolean owner;

    @Column(name = "like_count", nullable = false)
    private int likeCount = 0;

    @Column(name = "created_at", nullable = false)
    private LocalDateTime createdAt = LocalDateTime.now();

    @Column(name = "updated_at")
    private LocalDateTime updatedAt;
}

/*
 * [파일 설명]
 * - 장소에 대한 리뷰(텍스트 + 별점)를 나타내는 엔티티.
 * - owner=true이면 사장님이 직접 남긴 리뷰로 UI에서 "사장" 배지를 보여줄 수 있음.
 */
