package org.iclass.finalproject.store;

import org.iclass.finalproject.common.ApiResponse;
import org.iclass.finalproject.store.dto.StoreCreateRequest;
import org.iclass.finalproject.store.dto.StoreResponse;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.stream.Collectors;

@RestController
@RequestMapping("/api/stores")
@CrossOrigin(origins = "http://localhost:5173")
public class StoreController {

    private final StoreRepository storeRepository;

    public StoreController(StoreRepository storeRepository) {
        this.storeRepository = storeRepository;
    }

    // 전체 가게 목록 (지도에 뿌릴 용도)
    @GetMapping
    public ApiResponse<List<StoreResponse>> list() {
        List<Store> stores = storeRepository.findAll();
        List<StoreResponse> result = stores.stream()
                .map(StoreResponse::from)
                .collect(Collectors.toList());
        return ApiResponse.ok(result);
    }

    // 가게 등록
    @PostMapping
    public ApiResponse<StoreResponse> create(@RequestBody StoreCreateRequest req) {
        Store store = new Store();
        store.setStoreName(req.getStoreName());
        store.setFoodTypeId(req.getFoodTypeId());
        store.setStoreAddress(req.getStoreAddress());
        store.setLat(req.getLat());
        store.setLng(req.getLng());

        Store saved = storeRepository.save(store);
        return ApiResponse.ok(StoreResponse.from(saved));
    }
}
