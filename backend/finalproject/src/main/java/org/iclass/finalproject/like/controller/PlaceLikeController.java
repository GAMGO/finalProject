package org.iclass.finalproject.like.controller;

import lombok.RequiredArgsConstructor;
import org.iclass.finalproject.common.ApiResponse;
import org.iclass.finalproject.common.CurrentUser;
import org.iclass.finalproject.like.service.PlaceLikeService;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/places")
@RequiredArgsConstructor
public class PlaceLikeController {

    private final PlaceLikeService placeLikeService;

    @PostMapping("/{id}/like")
    public ApiResponse<Void> like(@PathVariable Long id) {
        Long userId = CurrentUser.getUserId();
        placeLikeService.likePlace(userId, id);
        return ApiResponse.ok(null);
    }

    @DeleteMapping("/{id}/like")
    public ApiResponse<Void> unlike(@PathVariable Long id) {
        Long userId = CurrentUser.getUserId();
        placeLikeService.unlikePlace(userId, id);
        return ApiResponse.ok(null);
    }
}

/*
 * [파일 설명]
 * - 가게 좋아요/좋아요 해제 REST 엔드포인트.
 * - 인스타/유튜브처럼 하트 버튼 UI와 연결되는 부분.
 */
