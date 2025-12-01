package org.iclass.dish;

import jakarta.persistence.*;
import jakarta.validation.constraints.*;
import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
@Entity
@Table(name = "dish")
public class Dish {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long idx;

    @NotBlank
    @Size(max = 100)
    private String name;

    @Size(max = 1000)
    private String description;

    @Positive
    private int price; // 가장 작은 화폐단위(예: KRW 원) 기준

    protected Dish() {}

    public Dish(String name, String description, int price) {
        this.name = name;
        this.description = description;
        this.price = price;
    }
}
