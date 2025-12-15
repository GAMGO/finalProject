package org.iclass.store.service;

import lombok.RequiredArgsConstructor;
import org.iclass.store.dto.StoreChangeRequestResponse;
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
public class StoreChangeAdminService {

    private final StoreChangeRequestRepository changeRepo;
    private final StoreRepository storeRepo;

    // ✅ 스샷의 adminService.listPending() 빨간줄 해결
    @Transactional(readOnly = true)
    public List<StoreChangeRequestResponse> listPending() {
        return changeRepo.findByStatus(StoreChangeStatus.PENDING)
                .stream()
                .map(StoreChangeRequestResponse::from)
                .toList();
    }

    @Transactional
    public void approveChange(Long changeId, Long adminId) {
        StoreChangeRequest req = changeRepo.findById(changeId)
                .orElseThrow(() -> new IllegalArgumentException("변경 요청이 존재하지 않습니다: " + changeId));

        if (req.getStatus() != StoreChangeStatus.PENDING) return;

        req.setStatus(StoreChangeStatus.APPROVED);
        req.setReviewedBy(adminId);
        req.setReviewedAt(LocalDateTime.now());
        req.setRejectReason(null);

        if (req.getType() == StoreChangeType.CREATE) {
            Store s = new Store();
            s.setStoreName(req.getNewStoreName());
            s.setStoreAddress(req.getNewStoreAddress());
            s.setLat(req.getNewLat());
            s.setLng(req.getNewLng());
            s.setFoodTypeId(req.getNewFoodTypeId());

            s.setIsDeleted(false);
            s.setDeletedAt(null);

            storeRepo.save(s);

        } else if (req.getType() == StoreChangeType.UPDATE) {
            if (req.getStore() == null) {
                throw new IllegalStateException("UPDATE 요청인데 store가 null 입니다.");
            }

            Store store = storeRepo.findById(req.getStore().getIdx())
                    .orElseThrow(() -> new IllegalArgumentException("대상 Store가 없습니다: " + req.getStore().getIdx()));

            if (Boolean.TRUE.equals(store.getIsDeleted())) {
                throw new IllegalStateException("삭제된 노점은 수정할 수 없습니다.");
            }

            if (req.getNewStoreName() != null) store.setStoreName(req.getNewStoreName());
            if (req.getNewStoreAddress() != null) store.setStoreAddress(req.getNewStoreAddress());
            if (req.getNewLat() != null) store.setLat(req.getNewLat());
            if (req.getNewLng() != null) store.setLng(req.getNewLng());
            if (req.getNewFoodTypeId() != null) store.setFoodTypeId(req.getNewFoodTypeId());

            storeRepo.save(store);

        } else if (req.getType() == StoreChangeType.DELETE) {
            if (req.getStore() == null) {
                throw new IllegalStateException("DELETE 요청인데 store가 null 입니다.");
            }

            Store store = storeRepo.findById(req.getStore().getIdx())
                    .orElseThrow(() -> new IllegalArgumentException("대상 Store가 없습니다: " + req.getStore().getIdx()));

            store.setIsDeleted(true);
            store.setDeletedAt(LocalDateTime.now());
            storeRepo.save(store);
        }

        changeRepo.save(req);
    }

    @Transactional
    public void rejectChange(Long changeId, Long adminId, String reason) {
        StoreChangeRequest req = changeRepo.findById(changeId)
                .orElseThrow(() -> new IllegalArgumentException("변경 요청이 존재하지 않습니다: " + changeId));

        if (req.getStatus() != StoreChangeStatus.PENDING) return;

        req.setStatus(StoreChangeStatus.REJECTED);
        req.setReviewedBy(adminId);
        req.setReviewedAt(LocalDateTime.now());
        req.setRejectReason((reason == null || reason.isBlank()) ? "사유 없음" : reason);

        changeRepo.save(req);
    }
}
