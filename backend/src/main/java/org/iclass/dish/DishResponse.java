package org.iclass.dish;

public class DishResponse {
    private Long idx;
    private String name;
    private String description;
    private int price;

    public DishResponse(Long idx, String name, String description, int price) {
        this.idx = idx;
        this.name = name;
        this.description = description;
        this.price = price;
    }

    public Long getIdx() { return idx; }
    public String getName() { return name; }
    public String getDescription() { return description; }
    public int getPrice() { return price; }

    public static DishResponse from(Dish dish) {
        return new DishResponse(dish.getIdx(), dish.getName(), dish.getDescription(), dish.getPrice());
    }
}
