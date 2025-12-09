package org.iclass.favorite.dto;

public class FavoriteResponse {

    private Long idx;
    private Long favoriteStoreIdx;   // 🔥 추가

    private String category;
    private String title;
    private String address;
    private String note;
    private Double rating;
    private String imageUrl;
    private String videoUrl;

    // ---- getter / setter ----

    public Long getIdx() {
        return idx;
    }

    public void setIdx(Long idx) {
        this.idx = idx;
    }

    public Long getFavoriteStoreIdx() {
        return favoriteStoreIdx;
    }

    public void setFavoriteStoreIdx(Long favoriteStoreIdx) {
        this.favoriteStoreIdx = favoriteStoreIdx;
    }

    public String getCategory() {
        return category;
    }

    public void setCategory(String category) {
        this.category = category;
    }

    public String getTitle() {
        return title;
    }

    public void setTitle(String title) {
        this.title = title;
    }

    public String getAddress() {
        return address;
    }

    public void setAddress(String address) {
        this.address = address;
    }

    public String getNote() {
        return note;
    }

    public void setNote(String note) {
        this.note = note;
    }

    public Double getRating() {
        return rating;
    }

    public void setRating(Double rating) {
        this.rating = rating;
    }

    public String getImageUrl() {
        return imageUrl;
    }

    public void setImageUrl(String imageUrl) {
        this.imageUrl = imageUrl;
    }

    public String getVideoUrl() {
        return videoUrl;
    }

    public void setVideoUrl(String videoUrl) {
        this.videoUrl = videoUrl;
    }
}
