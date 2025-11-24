package org.iclass.finalproject.dish;

import jakarta.persistence.*;
import jakarta.validation.constraints.*;

@Entity
@Table(name = "dish")
public class Dish {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Size(max = 100)
    @Column(nullable = true)    // 명시적 null 허용(DDL 생성/ 동기화시 반영)
    private String name;

    @Positive
    @Column(nullable = true)
    private Integer price; // null 허용             

    protected Dish() {}

    public Dish(String name, String description, Integer price) {
        this.name = name;
        this.price = price;
    }

    public Long getId() { return id; }
    public String getName() { return name; }
    public Integer getPrice() { return price; }

    public void setName(String name) { this.name = name; }
    public void setPrice(Integer price) { this.price = price; }
}
