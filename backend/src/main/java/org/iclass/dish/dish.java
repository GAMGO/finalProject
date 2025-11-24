package org.iclass.dish;

import jakarta.persistence.*;
import jakarta.validation.constraints.*;

@Entity
@Table(name = "dish")
public class dish {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @NotBlank
    @Size(max = 100)
    private String name;

    @Size(max = 1000)
    private String description;

    @Positive
    private int price; // 가장 작은 화폐단위(예: KRW 원) 기준

    protected dish() {}

    public dish(String name, String description, int price) {
        this.name = name;
        this.description = description;
        this.price = price;
    }

    public Long getId() { return id; }
    public String getName() { return name; }
    public String getDescription() { return description; }
    public int getPrice() { return price; }

    public void setName(String name) { this.name = name; }
    public void setDescription(String description) { this.description = description; }
    public void setPrice(int price) { this.price = price; }
}
