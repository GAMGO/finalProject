package org.iclass.favorite.service;

import org.iclass.favorite.dto.FavoriteRequest;
import org.iclass.favorite.dto.FavoriteResponse;
import org.iclass.favorite.entity.FavoriteEntity;
import org.iclass.favorite.repository.FavoriteRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.NoSuchElementException;
import java.util.stream.Collectors;

@Service
public class FavoriteService {

    private final FavoriteRepository favoriteRepository;

    @Autowired
    public FavoriteService(FavoriteRepository favoriteRepository) {
        this.favoriteRepository = favoriteRepository;
    }

    // TODO: 나중에 팀원이 로그인 붙이면 여기만 수정
    private Long getCustomerIdx() {
        return 1L;  // 임시 하드코딩
    }

    public List<FavoriteResponse> getMyFavorites() {
        Long customerIdx = getCustomerIdx();

        // ☆ repository 메서드명도 같이 수정했음
        return favoriteRepository.findByCustomerIdxOrderByCreatedAtDesc(customerIdx)
                .stream()
                .map(this::toResponse)
                .collect(Collectors.toList());
    }

    public FavoriteResponse createFavorite(FavoriteRequest req) {
        FavoriteEntity entity = new FavoriteEntity();

        entity.setCustomerIdx(getCustomerIdx());

        // ☆ 카테고리 / 상호 / 주소 세팅
        entity.setCategory(req.getCategory());
        entity.setTitle(req.getTitle());

        // 프론트에서 favoriteAddress 로 보내고 있으므로 둘 중 있는 값 사용
        String addr = req.getFavoriteAddress();
        if (addr == null || addr.isBlank()) {
            addr = req.getAddress();
        }
        entity.setAddress(addr);

        entity.setNote(req.getNote());
        entity.setRating(req.getRating());
        entity.setImageUrl(req.getImageUrl());
        entity.setVideoUrl(req.getVideoUrl());

        FavoriteEntity saved = favoriteRepository.save(entity);
        return toResponse(saved);
    }

    public FavoriteResponse updateFavorite(Long idx, FavoriteRequest req) {
        FavoriteEntity entity = favoriteRepository.findById(idx)
                .orElseThrow(() -> new NoSuchElementException("즐겨찾기를 찾을 수 없음: " + idx));

        // ☆ 수정 시에도 카테고리 / 상호 / 주소 반영
        entity.setCategory(req.getCategory());
        entity.setTitle(req.getTitle());

        String addr = req.getFavoriteAddress();
        if (addr == null || addr.isBlank()) {
            addr = req.getAddress();
        }
        entity.setAddress(addr);

        entity.setNote(req.getNote());
        entity.setRating(req.getRating());
        entity.setImageUrl(req.getImageUrl());
        entity.setVideoUrl(req.getVideoUrl());

        FavoriteEntity saved = favoriteRepository.save(entity);
        return toResponse(saved);
    }

    public void deleteFavorite(Long idx) {
        favoriteRepository.deleteById(idx);
    }

    // 엔티티 -> 응답 DTO
    private FavoriteResponse toResponse(FavoriteEntity entity) {
        FavoriteResponse dto = new FavoriteResponse();
        dto.setIdx(entity.getIdx());
        dto.setCategory(entity.getCategory());   // ☆ 추가
        dto.setTitle(entity.getTitle());         // ☆ 추가
        dto.setAddress(entity.getAddress());     // ☆ 추가

        dto.setNote(entity.getNote());
        dto.setRating(entity.getRating());
        dto.setImageUrl(entity.getImageUrl());
        dto.setVideoUrl(entity.getVideoUrl());
        return dto;
    }
}
