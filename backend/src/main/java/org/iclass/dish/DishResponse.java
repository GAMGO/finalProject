package org.iclass.dish;

public class DishResponse {
    private Long id;
    private String name;
    private String description;
    private int price;

    public DishResponse(Long id, String name, String description, int price) {
        this.id = id;
        this.name = name;
        this.description = description;
        this.price = price;
    }

    public Long getId() { return id; }
    public String getName() { return name; }
    public String getDescription() { return description; }
    public int getPrice() { return price; }

    public static DishResponse from(dish dish) {
        return new DishResponse(dish.getId(), dish.getName(), dish.getDescription(), dish.getPrice());
    }
}
