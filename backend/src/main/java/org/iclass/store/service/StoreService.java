package org.iclass.store.service;

import jakarta.persistence.EntityNotFoundException;
import lombok.RequiredArgsConstructor;
import org.iclass.store.dto.StoreCreateRequest;
import org.iclass.store.dto.StoreUpdateRequest;
import org.iclass.store.dto.StoreResponse;
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
import java.util.List;

@Service
@RequiredArgsConstructor
@Transactional(readOnly = true)
public class StoreService {

    private final StoreRepository storeRepository;
    private final StoreChangeRequestRepository changeRepository;

    public List<StoreResponse> listStores() {
        return storeRepository.findAll()
                .stream()
                .map(StoreResponse::from)
                .toList();
    }

    private static final DateTimeFormatter HHMM = DateTimeFormatter.ofPattern("HH:mm");

    private String toTime5(LocalDateTime dt) {
        if (dt == null) return null;
        LocalTime t = dt.toLocalTime();
        return t.format(HHMM);
    }

    @Transactional
    public Long createStore(StoreCreateRequest req, Long ownerId) {
        Store store = new Store();

        // ✅ trim 방어 (검증은 @NotBlank/@NotNull이 먼저)
        store.setStoreName(req.getStoreName() == null ? null : req.getStoreName().trim());
        store.setStoreAddress(req.getStoreAddress() == null ? null : req.getStoreAddress().trim());

        store.setOpenTime(toTime5(req.getOpenTime()));
        store.setCloseTime(toTime5(req.getCloseTime()));

        store.setLat(req.getLat());
        store.setLng(req.getLng());

        // ✅ 핵심(필수)
        store.setFoodTypeId(req.getFoodTypeId());

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
                .newOpenTime(toTime5(req.getOpenTime()))
                .newCloseTime(toTime5(req.getCloseTime()))
                .newStoreAddress(req.getStoreAddress())
                .newLat(req.getLat())
                .newLng(req.getLng())
                // .newFoodTypeId(req.getFoodTypeId()) // 필요하면 StoreChangeRequest에 필드 추가
                .build();

        return changeRepository.save(change).getId();
    }

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
