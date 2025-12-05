package org.iclass.store.service;

import jakarta.persistence.EntityNotFoundException;
import lombok.RequiredArgsConstructor;

import org.iclass.store.enums.StoreChangeStatus;
import org.iclass.store.enums.StoreChangeType;
import org.iclass.store.entity.Store;
import org.iclass.store.entity.StoreChangeRequest;
import org.iclass.store.entity.StoreChangeRequestResponse;
import org.iclass.store.repository.StoreChangeRequestRepository;
import org.iclass.store.repository.StoreRepository;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.util.List;

@Service
@RequiredArgsConstructor
@Transactional(readOnly = true) 
public class StoreChangeAdminService {

    private final StoreRepository storeRepository;
    private final StoreChangeRequestRepository changeRepository;

    /**
     * 승인 대기 목록
     */
    public List<StoreChangeRequestResponse> listPending() {
        return changeRepository.findByStatus(StoreChangeStatus.PENDING)
                .stream()
                .map(StoreChangeRequestResponse::from)
                .toList();
    }

    /**
     * 변경 요청 승인
     */
    @Transactional
    public void approveChange(Long changeId, Long adminId) {
        StoreChangeRequest c = changeRepository.findById(changeId)
                .orElseThrow(() -> new EntityNotFoundException("변경 요청을 찾을 수 없습니다."));

        if (c.getStatus() != StoreChangeStatus.PENDING) {
            throw new IllegalStateException("이미 처리된 요청입니다.");
        }

        Store store = c.getStore();

        if (c.getType() == StoreChangeType.UPDATE) {
            // StoreChangeRequest의 변경 필드를 실제 Store 엔티티에 반영
            if (c.getNewStoreName() != null) {
                store.setStoreName(c.getNewStoreName());
            }
            if (c.getNewOpenTime() != null) {
                store.setOpenTime(c.getNewOpenTime());     // VARCHAR(5) 매핑된 String
            }
            if (c.getNewCloseTime() != null) {
                store.setCloseTime(c.getNewCloseTime());   // VARCHAR(5) 매핑된 String
            }
            if (c.getNewStoreAddress() != null) {
                store.setStoreAddress(c.getNewStoreAddress());
            }
      
            if (c.getNewLat() != null) {
                store.setLat(c.getNewLat());
            }
            if (c.getNewLng() != null) {
                store.setLng(c.getNewLng());
            }
            // JPA dirty checking으로 UPDATE 자동 반영

        } else if (c.getType() == StoreChangeType.DELETE) {
            storeRepository.delete(store);
        }

        c.setStatus(StoreChangeStatus.APPROVED);
        c.setReviewedBy(adminId);
        c.setReviewedAt(LocalDateTime.now());
    }

    /**
     * 변경 요청 거절
     */
    @Transactional
    public void rejectChange(Long changeId, Long adminId, String reason) {
        StoreChangeRequest c = changeRepository.findById(changeId)
                .orElseThrow(() -> new EntityNotFoundException("변경 요청을 찾을 수 없습니다."));

        if (c.getStatus() != StoreChangeStatus.PENDING) {
            throw new IllegalStateException("이미 처리된 요청입니다.");
        }

        c.setStatus(StoreChangeStatus.REJECTED);
        c.setReviewedBy(adminId);
        c.setReviewedAt(LocalDateTime.now());
        c.setRejectReason(reason);
    }
}
