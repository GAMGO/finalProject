package org.iclass.finalproject.place.model;

import jakarta.persistence.*;
import lombok.Getter;
import lombok.Setter;
import org.iclass.finalproject.user.model.User;

import java.time.LocalDateTime;

@Entity
@Table(name = "places")
@Getter @Setter
public class Place {

    @Id @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Enumerated(EnumType.STRING)
    @Column(nullable = false, length = 20)
    private PlaceStatus status;

    @Column(length = 100)
    private String name; // 상호 없으면 null

    @Enumerated(EnumType.STRING)
    @Column(nullable = false, length = 30)
    private PlaceCategory category;

    @Column(nullable = false, length = 255)
    private String address;

    @Column(nullable = false)
    private Double latitude;

    @Column(nullable = false)
    private Double longitude;

    @Column(name = "short_description", length = 255)
    private String shortDescription;

    @Column(name = "main_menu", length = 100)
    private String mainMenu;

    @Column(name = "price_level")
    private Integer priceLevel; // 1~4

    @Column(length = 255)
    private String tags; // "#혼밥,#퇴근길" 등

    @Column(name = "like_count", nullable = false)
    private Integer likeCount = 0;

    @Column(name = "avg_rating", nullable = false)
    private Double avgRating = 0.0;

    @Column(name = "review_count", nullable = false)
    private Integer reviewCount = 0;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "created_by", nullable = false)
    private User createdBy;

    @Column(name = "created_at", nullable = false, updatable = false)
    private LocalDateTime createdAt = LocalDateTime.now();

    @Column(name = "updated_at")
    private LocalDateTime updatedAt;
}

/*
 * [파일 설명]
 * - 유저들이 등록하는 음식점/포장마차/푸드트럭 정보를 담는 엔티티.
 * - 위치정보(위도/경도), 카테고리, 대표메뉴, 좋아요/평점 등의 정보를 보관.
 * - 리뷰/즐겨찾기/사장인증 등 여러 도메인이 이 Place를 중심으로 연결된다.
 */
