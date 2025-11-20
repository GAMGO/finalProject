package org.iclass.finalproject.place.controller;

import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.iclass.finalproject.common.ApiResponse;
import org.iclass.finalproject.common.CurrentUser;
import org.iclass.finalproject.favorite.service.FavoriteService;
import org.iclass.finalproject.place.dto.PlaceCreateRequest;
import org.iclass.finalproject.place.dto.PlaceResponse;
import org.iclass.finalproject.place.model.Place;
import org.iclass.finalproject.place.service.PlaceService;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/places")
@RequiredArgsConstructor
public class PlaceController {

    private final PlaceService placeService;
    private final FavoriteService favoriteService;

    @PostMapping
    public ApiResponse<Long> createPlace(@RequestBody @Valid PlaceCreateRequest req) {
        Long userId = CurrentUser.getUserId();
        Place p = placeService.createPlace(userId, req);
        return ApiResponse.ok(p.getId());
    }

    @GetMapping("/{id}")
    public ApiResponse<PlaceResponse> getPlace(@PathVariable Long id) {
        Long userId = CurrentUser.getUserId();
        Place p = placeService.getById(id);
        boolean favorite = favoriteService.isFavorite(userId, id);
        return ApiResponse.ok(placeService.toResponse(p, favorite));
    }

    @GetMapping
    public ApiResponse<List<PlaceResponse>> listPlaces() {
        Long userId = CurrentUser.getUserId();
        List<Place> places = placeService.listAll();
        var res = places.stream()
                .map(p -> placeService.toResponse(p, favoriteService.isFavorite(userId, p.getId())))
                .toList();
        return ApiResponse.ok(res);
    }
}

/*
 * [파일 설명]
 * - 장소 등록/조회 REST 엔드포인트.
 * - 리스트/상세에서 즐겨찾기 여부까지 함께 내려줘서 프론트가 바로 UI를 그릴 수 있게 해줌.
 */
