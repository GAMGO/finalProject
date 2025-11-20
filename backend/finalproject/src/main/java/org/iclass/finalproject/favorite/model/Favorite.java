package org.iclass.finalproject.favorite.model;

import jakarta.persistence.*;
import lombok.Getter;
import lombok.Setter;
import org.iclass.finalproject.place.model.Place;
import org.iclass.finalproject.user.model.User;

import java.time.LocalDateTime;

@Entity
@Table(name = "favorites",
       uniqueConstraints = @UniqueConstraint(columnNames = {"user_id", "place_id"}))
@Getter @Setter
public class Favorite {

    @Id @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @ManyToOne(fetch = FetchType.LAZY) @JoinColumn(name = "user_id", nullable = false)
    private User user;

    @ManyToOne(fetch = FetchType.LAZY) @JoinColumn(name = "place_id", nullable = false)
    private Place place;

    @Column(name = "created_at", nullable = false)
    private LocalDateTime createdAt = LocalDateTime.now();
}

/*
 * [파일 설명]
 * - "어떤 유저가 어떤 장소를 즐겨찾기했는지"를 나타내는 엔티티.
 * - user_id + place_id를 UNIQUE로 묶어 중복 즐겨찾기를 방지.
 */
