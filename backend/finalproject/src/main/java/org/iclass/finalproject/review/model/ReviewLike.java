package org.iclass.finalproject.review.model;

import jakarta.persistence.*;
import lombok.Getter;
import lombok.Setter;
import org.iclass.finalproject.user.model.User;

@Entity
@Table(name = "review_likes",
       uniqueConstraints = @UniqueConstraint(columnNames = {"review_id", "user_id"}))
@Getter @Setter
public class ReviewLike {

    @Id @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @ManyToOne(fetch = FetchType.LAZY) @JoinColumn(name = "review_id")
    private Review review;

    @ManyToOne(fetch = FetchType.LAZY) @JoinColumn(name = "user_id")
    private User user;
}

/*
 * [파일 설명]
 * - 리뷰에 달린 "좋아요"를 저장하는 엔티티.
 * - 좋아요 수에 따라 "대표 리뷰"를 뽑는 데 사용.
 */
