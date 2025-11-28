package org.iclass.dish;

import jakarta.persistence.*;
import jakarta.validation.constraints.*;
import lombok.Getter;
import lombok.Setter;
import lombok.NoArgsConstructor;
import lombok.AllArgsConstructor;
import lombok.AccessLevel;
import lombok.Builder;

@Entity
@Table(name = "dish")
@Getter
@Setter
@NoArgsConstructor(access = AccessLevel.PROTECTED) // JPA용 기본 생성자 보호
@AllArgsConstructor // 모든 필드 생성자 (필요 시)
@Builder
public class Dish {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long idx;

    @NotBlank
    @Size(max = 100)
    @Column(nullable = false, length = 100, unique = true) // 중복 방지 권장
    private String name;

    @Size(max = 1000)
    @Column(length = 1000)
    private String description;

    @Positive
    private int price; // 가장 작은 화폐 단위(예: KRW 원)
}
