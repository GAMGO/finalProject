package org.iclass.finalproject.owner.model;

import jakarta.persistence.*;
import lombok.Getter;
import lombok.Setter;
import org.iclass.finalproject.place.model.Place;
import org.iclass.finalproject.user.model.User;

import java.time.LocalDateTime;

@Entity
@Table(name = "owner_verification")
@Getter @Setter
public class OwnerVerification {

    @Id @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @ManyToOne(fetch = FetchType.LAZY) @JoinColumn(name = "place_id")
    private Place place;

    @ManyToOne(fetch = FetchType.LAZY) @JoinColumn(name = "user_id")
    private User user;

    @Enumerated(EnumType.STRING)
    @Column(nullable = false)
    private OwnerVerificationStatus status = OwnerVerificationStatus.PENDING;

    @Column(name = "requested_at", nullable = false)
    private LocalDateTime requestedAt = LocalDateTime.now();

    @Column(name = "verified_at")
    private LocalDateTime verifiedAt;

    private String memo;
}

/*
 * [파일 설명]
 * - 가게 사장님이 자신의 가게임을 인증 요청할 때 사용하는 엔티티.
 * - 관리자는 이 엔티티를 보고 승인/거절을 결정하고, 승인 시 Place.status를 VERIFIED로 변경.
 */
