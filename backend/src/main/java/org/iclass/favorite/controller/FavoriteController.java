package org.iclass.favorite.controller;

import org.iclass.favorite.dto.FavoriteRequest;
import org.iclass.favorite.dto.FavoriteResponse;
import org.iclass.favorite.service.FavoriteService;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/favorites")
@CrossOrigin(origins = "http://localhost:5173")
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
    public FavoriteResponse updateFavorite(@PathVariable Long idx,
                                           @RequestBody FavoriteRequest req) {
        return favoriteService.updateFavorite(idx, req);
    }

    @DeleteMapping("/{id}")
    public void deleteFavorite(@PathVariable Long idx) {
        favoriteService.deleteFavorite(idx);
    }
}
