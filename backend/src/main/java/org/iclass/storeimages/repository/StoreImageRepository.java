package org.iclass.storeimages.repository;

import org.springframework.data.jpa.repository.JpaRepository;
import org.iclass.storeimages.entity.*;

public interface StoreImageRepository extends JpaRepository<StoreImage, Long> {
}
