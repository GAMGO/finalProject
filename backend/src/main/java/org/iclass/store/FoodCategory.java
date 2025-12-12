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

    private final Long idx;
    private final String label;

    FoodCategory(Long idx, String label) {
        this.idx = idx;
        this.label = label;
    }

    public Long getIdx() { return idx; }
    public String getLabel() { return label; }

    public static String labelOf(Long idx) {
        FoodCategory c = of(idx);
        return c == null ? null : c.label;
    }

    // ✅ 추가
    public static FoodCategory of(Long idx) {
        if (idx == null) return null;
        for (FoodCategory c : values()) {
            if (c.idx.equals(idx)) return c;
        }
        return null;
    }
}
