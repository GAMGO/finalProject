package org.iclass.store.repository;

import org.iclass.store.enums.StoreChangeStatus;
import org.iclass.store.entity.StoreChangeRequest;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;

public interface StoreChangeRequestRepository
        extends JpaRepository<StoreChangeRequest, Long> {

    List<StoreChangeRequest> findByStatus(StoreChangeStatus status);
}
