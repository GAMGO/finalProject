package org.iclass.finalproject.dish;

public class DishResponse {
    private Long id;
    private String name;
    private String description;
    private int price;

    public DishResponse(Long id, String name, int price) {
        this.id = id;
        this.name = name;
        this.price = price;
    }

    public Long getId() { return id; }
    public String getName() { return name; }
    public String getDescription() { return description; }
    public int getPrice() { return price; }

    public static DishResponse from(Dish dish) {
        return new DishResponse(dish.getId(), dish.getName(), dish.getPrice());
    }
}
