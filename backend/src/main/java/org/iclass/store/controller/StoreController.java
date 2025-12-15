package org.iclass.store.controller;

import lombok.RequiredArgsConstructor;
import org.iclass.store.dto.StoreCreateRequest;
import org.iclass.store.dto.StoreResponse;
import org.iclass.store.dto.StoreUpdateRequest;
import org.iclass.store.service.StoreService;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/stores")
@RequiredArgsConstructor
public class StoreController {

    private final StoreService storeService;

    // ✅ 지도 목록
    @GetMapping
    public ResponseEntity<List<StoreResponse>> list() {
        return ResponseEntity.ok(storeService.listStores());
    }

    // ✅ 등록 요청 (PENDING 쌓임)
    @PostMapping
    public ResponseEntity<Long> create(@RequestBody StoreCreateRequest req) {
        Long ownerId = 1L; // TODO: 로그인 유저 ID로 교체
        Long changeId = storeService.createStore(req, ownerId);
        return ResponseEntity.ok(changeId);
    }

    // ✅ 수정 요청
    @PostMapping("/{storeIdx}/update-request")
    public ResponseEntity<Long> updateRequest(@PathVariable Long storeIdx,
                                              @RequestBody StoreUpdateRequest req) {
        Long requesterId = 1L; // TODO
        Long changeId = storeService.requestUpdateStore(storeIdx, requesterId, req);
        return ResponseEntity.ok(changeId);
    }

    // ✅ 삭제 요청
    @PostMapping("/{storeIdx}/delete-request")
    public ResponseEntity<Long> deleteRequest(@PathVariable Long storeIdx) {
        Long requesterId = 1L; // TODO
        Long changeId = storeService.requestDeleteStore(storeIdx, requesterId);
        return ResponseEntity.ok(changeId);
    }
}
