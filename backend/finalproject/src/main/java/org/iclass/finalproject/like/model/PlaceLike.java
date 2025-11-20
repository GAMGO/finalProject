package org.iclass.finalproject.like.model;

import jakarta.persistence.*;
import lombok.Getter;
import lombok.Setter;
import org.iclass.finalproject.place.model.Place;
import org.iclass.finalproject.user.model.User;

@Entity
@Table(name = "place_likes",
       uniqueConstraints = @UniqueConstraint(columnNames = {"user_id", "place_id"}))
@Getter @Setter
public class PlaceLike {

    @Id @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @ManyToOne(fetch = FetchType.LAZY) @JoinColumn(name = "user_id")
    private User user;

    @ManyToOne(fetch = FetchType.LAZY) @JoinColumn(name = "place_id")
    private Place place;
}

/*
 * [파일 설명]
 * - 특정 유저가 가게에 "좋아요"를 누른 기록을 저장하는 엔티티.
 * - 즐겨찾기와는 별개로, 전체 좋아요 수 집계에 사용.
 */
