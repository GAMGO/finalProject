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

    public Long getCustomer_id() { return id; }
    public String getName() { return name; }
    public String getDescription() { return description; }
    public int getPrice() { return price; }

    public static DishResponse from(Dish dish) {
        return new DishResponse(dish.getCustomer_id(), dish.getName(), dish.getDescription(), dish.getPrice());
    }
}
