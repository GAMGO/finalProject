package org.iclass.favorite.controller;

import org.iclass.favorite.dto.FavoriteRequest;
import org.iclass.favorite.dto.FavoriteResponse;
import org.iclass.favorite.service.FavoriteService;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/favorites")
public class FavoriteController {

    private final FavoriteService favoriteService;

    public FavoriteController(FavoriteService favoriteService) {
        this.favoriteService = favoriteService;
    }

    @GetMapping
    public List<FavoriteResponse> getFavorites() {
        // ApiResponse 안 쓰고 바로 리스트 리턴
        return favoriteService.getMyFavorites();
    }

    @PostMapping
    public FavoriteResponse createFavorite(@RequestBody FavoriteRequest req) {
        return favoriteService.createFavorite(req);
    }

    @PutMapping("/{id}")
    public FavoriteResponse updateFavorite(
            // ✅ URL 의 {id} 를 idx 파라미터에 매핑
            @PathVariable("id") Long idx,
            @RequestBody FavoriteRequest req
    ) {
        return favoriteService.updateFavorite(idx, req);
    }

    @DeleteMapping("/{id}")
    public void deleteFavorite(
            // ✅ 위와 동일하게 명시
            @PathVariable("id") Long idx
    ) {
        favoriteService.deleteFavorite(idx);
    }
}
