package org.iclass.finalproject.owner.service;

import lombok.RequiredArgsConstructor;
import org.iclass.finalproject.owner.model.OwnerVerification;
import org.iclass.finalproject.owner.model.OwnerVerificationStatus;
import org.iclass.finalproject.owner.repository.OwnerVerificationRepository;
import org.iclass.finalproject.place.model.Place;
import org.iclass.finalproject.place.model.PlaceStatus;
import org.iclass.finalproject.place.repository.PlaceRepository;
import org.iclass.finalproject.user.model.User;
import org.iclass.finalproject.user.repository.UserRepository;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.util.List;

@Service
@RequiredArgsConstructor
public class OwnerVerificationService {

    private final OwnerVerificationRepository ownerVerificationRepository;
    private final UserRepository userRepository;
    private final PlaceRepository placeRepository;

    @Transactional
    public OwnerVerification request(Long userId, Long placeId, String memo) {
        User u = userRepository.findById(userId).orElseThrow();
        Place p = placeRepository.findById(placeId).orElseThrow();

        OwnerVerification ov = new OwnerVerification();
        ov.setUser(u);
        ov.setPlace(p);
        ov.setMemo(memo);
        ov.setStatus(OwnerVerificationStatus.PENDING);

        return ownerVerificationRepository.save(ov);
    }

    @Transactional(readOnly = true)
    public List<OwnerVerification> listPending() {
        return ownerVerificationRepository.findByStatus(OwnerVerificationStatus.PENDING);
    }

    @Transactional
    public void approve(Long ovId) {
        OwnerVerification ov = ownerVerificationRepository.findById(ovId).orElseThrow();
        ov.setStatus(OwnerVerificationStatus.APPROVED);
        ov.setVerifiedAt(LocalDateTime.now());
        ownerVerificationRepository.save(ov);

        Place p = ov.getPlace();
        p.setStatus(PlaceStatus.VERIFIED);
        placeRepository.save(p);
    }

    @Transactional
    public void reject(Long ovId, String memo) {
        OwnerVerification ov = ownerVerificationRepository.findById(ovId).orElseThrow();
        ov.setStatus(OwnerVerificationStatus.REJECTED);
        ov.setMemo(memo);
        ownerVerificationRepository.save(ov);
    }
}

/*
 * [파일 설명]
 * - 사장 인증 요청 생성/조회/승인/거절 로직 담당 서비스.
 * - 승인 시 해당 Place의 상태를 VERIFIED로 바꿔서 "인증됨" 뱃지를 표시할 수 있게 함.
 */
