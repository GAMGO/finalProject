package org.iclass.finalproject.place.service;

import lombok.RequiredArgsConstructor;
import org.iclass.finalproject.place.dto.PlaceCreateRequest;
import org.iclass.finalproject.place.dto.PlaceResponse;
import org.iclass.finalproject.place.model.Place;
import org.iclass.finalproject.place.model.PlaceStatus;
import org.iclass.finalproject.place.repository.PlaceRepository;
import org.iclass.finalproject.user.model.User;
import org.iclass.finalproject.user.repository.UserRepository;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;

@Service
@RequiredArgsConstructor
public class PlaceService {

    private final PlaceRepository placeRepository;
    private final UserRepository userRepository;

    @Transactional
    public Place createPlace(Long userId, PlaceCreateRequest req) {
        User creator = userRepository.findById(userId)
                .orElseThrow(() -> new IllegalArgumentException("user not found"));

        Place p = new Place();
        p.setStatus(PlaceStatus.DISCOVERED);
        p.setCategory(req.getCategory());
        p.setName(req.getName());
        p.setAddress(req.getAddress());
        p.setLatitude(req.getLatitude());
        p.setLongitude(req.getLongitude());
        p.setShortDescription(req.getShortDescription());
        p.setMainMenu(req.getMainMenu());
        p.setPriceLevel(req.getPriceLevel());
        p.setTags(req.getTags());
        p.setCreatedBy(creator);

        return placeRepository.save(p);
    }

    @Transactional(readOnly = true)
    public Place getById(Long id) {
        return placeRepository.findById(id)
                .orElseThrow(() -> new IllegalArgumentException("place not found"));
    }

    @Transactional(readOnly = true)
    public List<Place> listAll() {
        return placeRepository.findAll();
    }

    public PlaceResponse toResponse(Place p, boolean favorite) {
        return PlaceResponse.builder()
                .id(p.getId())
                .status(p.getStatus())
                .name(p.getName())
                .category(p.getCategory())
                .address(p.getAddress())
                .latitude(p.getLatitude())
                .longitude(p.getLongitude())
                .shortDescription(p.getShortDescription())
                .mainMenu(p.getMainMenu())
                .priceLevel(p.getPriceLevel())
                .tags(p.getTags())
                .likeCount(p.getLikeCount())
                .avgRating(p.getAvgRating())
                .reviewCount(p.getReviewCount())
                .favorite(favorite)
                .build();
    }
}

/*
 * [파일 설명]
 * - 장소 등록/조회 로직 담당 서비스.
 * - 엔티티를 PlaceResponse로 변환하는 헬퍼(toResponse)도 이곳에서 담당.
 */
