package org.iclass.finalproject.owner.repository;

import org.iclass.finalproject.owner.model.OwnerVerification;
import org.iclass.finalproject.owner.model.OwnerVerificationStatus;
import org.iclass.finalproject.place.model.Place;
import org.iclass.finalproject.user.model.User;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;
import java.util.Optional;

public interface OwnerVerificationRepository extends JpaRepository<OwnerVerification, Long> {

    Optional<OwnerVerification> findByPlaceAndUserAndStatus(Place place, User user,
                                                             OwnerVerificationStatus status);

    List<OwnerVerification> findByStatus(OwnerVerificationStatus status);
}

/*
 * [파일 설명]
 * - 사장 인증 요청(OwnerVerification) 조회/저장을 담당하는 Repository.
 * - 특정 가게/유저/상태 조합 확인 및 PENDING 목록 조회에 사용.
 */
