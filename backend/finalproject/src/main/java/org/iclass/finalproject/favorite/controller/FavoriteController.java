package org.iclass.finalproject.favorite.controller;

import lombok.RequiredArgsConstructor;
import org.iclass.finalproject.common.ApiResponse;
import org.iclass.finalproject.common.CurrentUser;
import org.iclass.finalproject.favorite.model.Favorite;
import org.iclass.finalproject.favorite.service.FavoriteService;
import org.iclass.finalproject.place.dto.PlaceResponse;
import org.iclass.finalproject.place.service.PlaceService;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api")
@RequiredArgsConstructor
public class FavoriteController {

    private final FavoriteService favoriteService;
    private final PlaceService placeService;

    @PostMapping("/places/{id}/favorite")
    public ApiResponse<Void> addFavorite(@PathVariable Long id) {
        Long userId = CurrentUser.getUserId();
        favoriteService.addFavorite(userId, id);
        return ApiResponse.ok(null);
    }

    @DeleteMapping("/places/{id}/favorite")
    public ApiResponse<Void> removeFavorite(@PathVariable Long id) {
        Long userId = CurrentUser.getUserId();
        favoriteService.removeFavorite(userId, id);
        return ApiResponse.ok(null);
    }

    @GetMapping("/me/favorites")
    public ApiResponse<List<PlaceResponse>> myFavorites() {
        Long userId = CurrentUser.getUserId();
        List<Favorite> favorites = favoriteService.getFavorites(userId);
        var res = favorites.stream()
                .map(f -> placeService.toResponse(f.getPlace(), true))
                .toList();
        return ApiResponse.ok(res);
    }
}

/*
 * [파일 설명]
 * - 즐겨찾기 관련 REST 엔드포인트.
 * - 장소 카드의 "북마크 버튼"과 설정/마이페이지의 "즐겨찾기 목록" 화면에서 호출.
 */
