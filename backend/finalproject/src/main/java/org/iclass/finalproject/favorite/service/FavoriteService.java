package org.iclass.finalproject.favorite.service;

import lombok.RequiredArgsConstructor;
import org.iclass.finalproject.favorite.model.Favorite;
import org.iclass.finalproject.favorite.repository.FavoriteRepository;
import org.iclass.finalproject.place.model.Place;
import org.iclass.finalproject.place.repository.PlaceRepository;
import org.iclass.finalproject.user.model.User;
import org.iclass.finalproject.user.repository.UserRepository;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;

@Service
@RequiredArgsConstructor
public class FavoriteService {

    private final FavoriteRepository favoriteRepository;
    private final UserRepository userRepository;
    private final PlaceRepository placeRepository;

    @Transactional
    public void addFavorite(Long userId, Long placeId) {
        User u = userRepository.findById(userId).orElseThrow();
        Place p = placeRepository.findById(placeId).orElseThrow();

        favoriteRepository.findByUserAndPlace(u, p)
                .ifPresent(f -> { throw new IllegalStateException("already favorite"); });

        Favorite f = new Favorite();
        f.setUser(u);
        f.setPlace(p);
        favoriteRepository.save(f);
    }

    @Transactional
    public void removeFavorite(Long userId, Long placeId) {
        User u = userRepository.findById(userId).orElseThrow();
        Place p = placeRepository.findById(placeId).orElseThrow();

        favoriteRepository.findByUserAndPlace(u, p)
                .ifPresent(favoriteRepository::delete);
    }

    @Transactional(readOnly = true)
    public boolean isFavorite(Long userId, Long placeId) {
        User u = userRepository.findById(userId).orElseThrow();
        Place p = placeRepository.findById(placeId).orElseThrow();
        return favoriteRepository.findByUserAndPlace(u, p).isPresent();
    }

    @Transactional(readOnly = true)
    public List<Favorite> getFavorites(Long userId) {
        User u = userRepository.findById(userId).orElseThrow();
        return favoriteRepository.findByUser(u);
    }
}

/*
 * [파일 설명]
 * - 즐겨찾기 추가/삭제/조회 비즈니스 로직.
 * - PlaceService와 함께 써서 "내 즐겨찾기 장소 리스트"를 구성하는 데 사용.
 */
