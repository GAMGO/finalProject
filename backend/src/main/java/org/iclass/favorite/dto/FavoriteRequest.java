package org.iclass.favorite.dto;

public class FavoriteRequest {

    private Long idx;
    private String category;
    private String title;

    // 프론트 쪽 payload 에서는 favoriteAddress 로 보내고 있으므로 둘 다 받자
    private String address;          // 예전/다른 코드용
    private String favoriteAddress;  // FavoritePage 에서 보내는 필드

    private String note;
    private Double rating;
    private String imageUrl;
    private String videoUrl;

    // ---- getter / setter ----

    // ☆ 오타 수정: getIDx -> getIdx
    public Long getIdx() {
        return idx;
    }

    public void setIdx(Long idx) {
        this.idx = idx;
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

    // address / favoriteAddress 둘 다 지원
    public String getAddress() {
        return address;
    }

    public void setAddress(String address) {
        this.address = address;
    }

    public String getFavoriteAddress() {
        return favoriteAddress;
    }

    public void setFavoriteAddress(String favoriteAddress) {
        this.favoriteAddress = favoriteAddress;
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
