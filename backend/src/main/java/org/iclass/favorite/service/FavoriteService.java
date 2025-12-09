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

        return favoriteRepository.findByCustomerIdxOrderByCreatedAtDesc(customerIdx)
                .stream()
                .map(this::toResponse)
                .collect(Collectors.toList());
    }

    public FavoriteResponse createFavorite(FavoriteRequest req) {
        FavoriteEntity entity = new FavoriteEntity();

        entity.setCustomerIdx(getCustomerIdx());

        // 🔥 노점 PK 저장 (지도에서 넘어오는 값)
        entity.setFavoriteStoreIdx(req.getFavoriteStoreIdx());

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

    public FavoriteResponse updateFavorite(Long idx, FavoriteRequest req) {
        FavoriteEntity entity = favoriteRepository.findById(idx)
                .orElseThrow(() -> new NoSuchElementException("즐겨찾기를 찾을 수 없음: " + idx));

        // 🔥 필요하면 수정도 가능
        entity.setFavoriteStoreIdx(req.getFavoriteStoreIdx());

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
        dto.setFavoriteStoreIdx(entity.getFavoriteStoreIdx());  // 🔥 추가

        dto.setCategory(entity.getCategory());
        dto.setTitle(entity.getTitle());
        dto.setAddress(entity.getAddress());

        dto.setNote(entity.getNote());
        dto.setRating(entity.getRating());
        dto.setImageUrl(entity.getImageUrl());
        dto.setVideoUrl(entity.getVideoUrl());
        return dto;
    }
}
