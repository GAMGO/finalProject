package org.iclass.store.controller;

import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.iclass.store.dto.StoreCreateRequest;
import org.iclass.store.dto.StoreUpdateRequest;
import org.iclass.store.dto.StoreResponse;
import org.iclass.store.service.StoreService;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/stores")
@RequiredArgsConstructor
public class StoreController {

    private final StoreService storeService;

    // ✅ 0) 점포 전체 목록 조회 (카카오맵에서 사용)
    @GetMapping
    public ResponseEntity<List<StoreResponse>> list() {
        return ResponseEntity.ok(storeService.listStores());
    }

    // 1) 점포 최초 등록: 바로 반영
    @PostMapping
    public ResponseEntity<Long> create(@Valid @RequestBody StoreCreateRequest req) {
        Long ownerId = 1L; // TODO: 로그인 유저 ID에서 가져오기
        Long id = storeService.createStore(req, ownerId);
        return ResponseEntity.ok(id);
    }

    // 2) 점포 수정 "요청"
    @PostMapping("/{storeIdx}/update-request")
    public ResponseEntity<Long> requestUpdate(@PathVariable Long storeIdx,
                                              @Valid @RequestBody StoreUpdateRequest req) {
        Long requesterId = 1L; // TODO: 로그인 유저 ID
        Long changeId = storeService.requestUpdateStore(storeIdx, requesterId, req);
        return ResponseEntity.ok(changeId);
    }

    // 3) 점포 삭제 "요청"
    @PostMapping("/{storeIdx}/delete-request")
    public ResponseEntity<Long> requestDelete(@PathVariable Long storeIdx) {
        Long requesterId = 1L; // TODO: 로그인 유저 ID
        Long changeId = storeService.requestDeleteStore(storeIdx, requesterId);
        return ResponseEntity.ok(changeId);
    }
}
