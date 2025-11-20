package org.iclass.finalproject.like.service;

import lombok.RequiredArgsConstructor;
import org.iclass.finalproject.like.model.PlaceLike;
import org.iclass.finalproject.like.repository.PlaceLikeRepository;
import org.iclass.finalproject.place.model.Place;
import org.iclass.finalproject.place.repository.PlaceRepository;
import org.iclass.finalproject.user.model.User;
import org.iclass.finalproject.user.repository.UserRepository;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

@Service
@RequiredArgsConstructor
public class PlaceLikeService {

    private final PlaceLikeRepository placeLikeRepository;
    private final UserRepository userRepository;
    private final PlaceRepository placeRepository;

    @Transactional
    public void likePlace(Long userId, Long placeId) {
        User u = userRepository.findById(userId).orElseThrow();
        Place p = placeRepository.findById(placeId).orElseThrow();

        if (placeLikeRepository.findByUserAndPlace(u, p).isPresent()) return;

        PlaceLike like = new PlaceLike();
        like.setUser(u);
        like.setPlace(p);
        placeLikeRepository.save(like);

        p.setLikeCount((int) placeLikeRepository.countByPlace(p));
        placeRepository.save(p);
    }

    @Transactional
    public void unlikePlace(Long userId, Long placeId) {
        User u = userRepository.findById(userId).orElseThrow();
        Place p = placeRepository.findById(placeId).orElseThrow();

        placeLikeRepository.findByUserAndPlace(u, p)
                .ifPresent(placeLikeRepository::delete);

        p.setLikeCount((int) placeLikeRepository.countByPlace(p));
        placeRepository.save(p);
    }
}

/*
 * [파일 설명]
 * - 가게 좋아요/좋아요 취소 로직 및 좋아요 수 캐싱 업데이트 담당.
 * - Place.likeCount 컬럼을 항상 실제 DB 카운트와 맞춰주는 역할.
 */
