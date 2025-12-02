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
import java.time.LocalTime;
import java.time.format.DateTimeFormatter;

@Service
@RequiredArgsConstructor
@Transactional(readOnly = true)
public class StoreService {

    private final StoreRepository storeRepository;
    private final StoreChangeRequestRepository changeRepository;

    // "HH:mm" 변환 유틸 (DTO가 LocalDateTime 을 주는 경우 대응)
    private static final DateTimeFormatter HHMM = DateTimeFormatter.ofPattern("HH:mm");
    private String toTime5(LocalDateTime dt) {
        if (dt == null) return null;
        LocalTime t = dt.toLocalTime();
        return t.format(HHMM); // e.g. "09:30"
    }

    /**
     * 점포 최초 등록: 바로 반영
     * StoreCreateRequest 필드: storeName, openTime(LocalDateTime), closeTime(LocalDateTime),
     *                          storeAddress, lat, lng
     *  - foodTypeId 는 스키마 변경으로 제거됨
     */
    @Transactional
    public Long createStore(StoreCreateRequest req, Long ownerId) {
        Store store = new Store();

        store.setStoreName(req.getStoreName());
        store.setOpenTime(toTime5(req.getOpenTime()));     // "HH:mm" 문자열로 저장
        store.setCloseTime(toTime5(req.getCloseTime()));   // "
        store.setStoreAddress(req.getStoreAddress());
        store.setLat(req.getLat());
        store.setLng(req.getLng());

        return storeRepository.save(store).getIdx();
    }

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
                .newOpenTime(toTime5(req.getOpenTime()))     // String("HH:mm") 필요
                .newCloseTime(toTime5(req.getCloseTime()))   // "
                .newStoreAddress(req.getStoreAddress())
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
