package org.iclass.storeimages.entity;

import org.iclass.customer.entity.CustomersEntity;
import org.iclass.store.entity.Store;
import jakarta.persistence.*;
import lombok.*;
import java.time.LocalDateTime;

@Getter @Setter @NoArgsConstructor @AllArgsConstructor @Builder
@Entity
@Table(name = "store_images",
       indexes = {
         @Index(name = "ix_si_store", columnList = "store_idx"),
         @Index(name = "ix_si_user", columnList = "upload_user_idx")
       })
public class StoreImage {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name = "idx")
    private Long idx;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "store_idx", foreignKey = @ForeignKey(name = "fk_si_store"))
    private Store store;

    @Column(name = "uploaded_at")
    private LocalDateTime uploadedAt;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "upload_user_idx", foreignKey = @ForeignKey(name = "fk_si_user"))
    private CustomersEntity uploadUser;

    @Column(name = "img_location", length = 1024)
    private String imgLocation;

    @Column(name = "img_type", length = 125)
    private String imgType;
}
