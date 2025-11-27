package org.iclass.store;

public enum FoodCategory {

    CHICKEN(1L, "통닭"),
    TAKOYAKI(2L, "타코야끼"),
    SOONDAE_GOPCHANG(3L, "순대곱창"),
    BUNGEOPPANG(4L, "붕어빵"),
    ROASTED_CHESTNUT(5L, "군밤/고구마"),
    CHICKEN_SKEWER(6L, "닭꼬치"),
    BUNSIK(7L, "분식"),
    SEAFOOD(8L, "해산물"),
    PUFFED_RICE(9L, "뻥튀기"),
    EGG_BREAD(10L, "계란빵"),
    CORN(11L, "옥수수");

    private final Long id;
    private final String label;

    FoodCategory(Long id, String label) {
        this.id = id;
        this.label = label;
    }

    public Long getCustomer_id() {
        return id;
    }

    public String getLabel() {
        return label;
    }

    public static String labelOf(Long id) {
        if (id == null) return null;
        for (FoodCategory c : values()) {
            if (c.id.equals(id)) return c.label;
        }
        return null;
    }
}
