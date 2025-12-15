package org.iclass.store.service;

import lombok.RequiredArgsConstructor;
import org.iclass.store.dto.StoreCreateRequest;
import org.iclass.store.dto.StoreResponse;
import org.iclass.store.dto.StoreUpdateRequest;
import org.iclass.store.entity.Store;
import org.iclass.store.entity.StoreChangeRequest;
import org.iclass.store.enums.StoreChangeStatus;
import org.iclass.store.enums.StoreChangeType;
import org.iclass.store.repository.StoreChangeRequestRepository;
import org.iclass.store.repository.StoreRepository;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.util.List;

@Service
@RequiredArgsConstructor
public class StoreService {

    private final StoreRepository storeRepo;
    private final StoreChangeRequestRepository changeRepo;

    // ✅ 지도 목록 (삭제 제외)
    @Transactional(readOnly = true)
    public List<StoreResponse> listStores() {
        return storeRepo.findAllByIsDeletedFalse()
                .stream()
                .map(StoreResponse::fromEntity)
                .toList();
    }

    // ✅ (프론트/컨트롤러에서 createStore를 부르는 경우 많아서 이 이름으로 제공)
    // "등록 요청" → StoreChangeRequest에 쌓고 admin 승인으로 stores 반영
    @Transactional
    public Long createStore(StoreCreateRequest req, Long requesterId) {
        StoreChangeRequest cr = new StoreChangeRequest();
        cr.setType(StoreChangeType.CREATE);
        cr.setStatus(StoreChangeStatus.PENDING);
        cr.setRequestedBy(requesterId);
        cr.setRequestedAt(LocalDateTime.now());
        cr.setStore(null);

        cr.setNewStoreName(req.getStoreName());
        cr.setNewStoreAddress(req.getStoreAddress());
        cr.setNewLat(req.getLat());
        cr.setNewLng(req.getLng());
        cr.setNewFoodTypeId(req.getFoodTypeId());

        return changeRepo.save(cr).getId();
    }

    @Transactional
    public Long requestUpdateStore(Long storeIdx, Long requesterId, StoreUpdateRequest req) {
        Store store = storeRepo.findById(storeIdx)
                .orElseThrow(() -> new IllegalArgumentException("Store가 없습니다: " + storeIdx));

        if (Boolean.TRUE.equals(store.getIsDeleted())) {
            throw new IllegalStateException("삭제된 노점은 수정 요청이 불가합니다.");
        }

        StoreChangeRequest cr = new StoreChangeRequest();
        cr.setType(StoreChangeType.UPDATE);
        cr.setStatus(StoreChangeStatus.PENDING);
        cr.setRequestedBy(requesterId);
        cr.setRequestedAt(LocalDateTime.now());
        cr.setStore(store);

        cr.setNewStoreName(req.getStoreName());
        cr.setNewStoreAddress(req.getStoreAddress());
        cr.setNewLat(req.getLat());
        cr.setNewLng(req.getLng());
        cr.setNewFoodTypeId(req.getFoodTypeId());

        return changeRepo.save(cr).getId();
    }

    @Transactional
    public Long requestDeleteStore(Long storeIdx, Long requesterId) {
        Store store = storeRepo.findById(storeIdx)
                .orElseThrow(() -> new IllegalArgumentException("Store가 없습니다: " + storeIdx));

        if (Boolean.TRUE.equals(store.getIsDeleted())) {
            throw new IllegalStateException("이미 삭제된 노점입니다.");
        }

        StoreChangeRequest cr = new StoreChangeRequest();
        cr.setType(StoreChangeType.DELETE);
        cr.setStatus(StoreChangeStatus.PENDING);
        cr.setRequestedBy(requesterId);
        cr.setRequestedAt(LocalDateTime.now());
        cr.setStore(store);

        return changeRepo.save(cr).getId();
    }
}
