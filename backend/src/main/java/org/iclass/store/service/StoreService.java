package org.iclass.store.service;

import jakarta.persistence.EntityNotFoundException;
import lombok.RequiredArgsConstructor;
import org.iclass.store.dto.StoreCreateRequest;
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

@Service
@RequiredArgsConstructor
@Transactional(readOnly = true)
public class StoreService {

    private final StoreRepository storeRepository;
    private final StoreChangeRequestRepository changeRepository;

    /**
     * 점포 최초 등록: 바로 반영
     * StoreCreateRequest는 Store 엔티티와 동일한 필드명을 가진다고 가정:
     *  - storeName, openTime, closeTime, storeAddress, foodTypeId, lat, lng
     */
    @Transactional
    public Long createStore(StoreCreateRequest req, Long ownerId) {
        Store store = new Store();

        store.setStoreName(req.getStoreName());
        store.setOpenTime(req.getOpenTime());
        store.setCloseTime(req.getCloseTime());
        store.setStoreAddress(req.getStoreAddress());
        store.setFoodTypeId(req.getFoodTypeId());
        store.setLat(req.getLat());
        store.setLng(req.getLng());

        // ownerId 등은 현재 Store 엔티티에 없으므로 사용 X (필요하면 필드 추가)

        return storeRepository.save(store).getIdx();
    }

    /**
     * 점포 수정 요청: 실제 Store는 건드리지 않고 변경 요청만 저장
     */
    @Transactional
    public Long requestUpdateStore(Long storeIdx, Long requesterId, StoreUpdateRequest req) {
        Store store = storeRepository.findById(storeIdx)
                .orElseThrow(() -> new EntityNotFoundException("가게 정보를 찾을 수 없습니다."));

        StoreChangeRequest change = StoreChangeRequest.builder()
                .store(store)
                .type(StoreChangeType.UPDATE)
                .status(StoreChangeStatus.PENDING)
                .requestedBy(requesterId)
                .requestedAt(LocalDateTime.now())
                .newStoreName(req.getStoreName())
                .newOpenTime(req.getOpenTime())
                .newCloseTime(req.getCloseTime())
                .newStoreAddress(req.getStoreAddress())
                .newFoodTypeId(req.getFoodTypeId())
                .newLat(req.getLat())
                .newLng(req.getLng())
                .build();

        return changeRepository.save(change).getId();
    }

    /**
     * 점포 삭제 요청: 바로 삭제하지 않고 승인 대기만 생성
     */
    @Transactional
    public Long requestDeleteStore(Long storeIdx, Long requesterId) {
        Store store = storeRepository.findById(storeIdx)
                .orElseThrow(() -> new EntityNotFoundException("가게 정보를 찾을 수 없습니다."));

        StoreChangeRequest change = StoreChangeRequest.builder()
                .store(store)
                .type(StoreChangeType.DELETE)
                .status(StoreChangeStatus.PENDING)
                .requestedBy(requesterId)
                .requestedAt(LocalDateTime.now())
                .build();

        return changeRepository.save(change).getId();
    }
}
