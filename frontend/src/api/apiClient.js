// 기존에 만들어 둔 axios 인스턴스
import axios from "axios";

const apiClient = axios.create({
  baseURL: "http://localhost:8080/",
});

// 즐겨찾기 API
export const favoriteApi = {
  async getAll() {
    const res = await apiClient.get('/favorites');
    // 백엔드에서 ApiResponse<T> 쓰면 보통 { status, message, data } 구조일 거라서
    return res.data.data ?? res.data;  // 둘 중 프로젝트 구조에 맞는 쪽으로 쓰이면 된다.
  },

  async create(favorite) {
    const res = await apiClient.post('/favorites', favorite);
    return res.data.data ?? res.data;
  },

  async update(id, favorite) {
    const res = await apiClient.put(`/favorites/${id}`, favorite);
    return res.data.data ?? res.data;
  },

  async remove(id) {
    await apiClient.delete(`/favorites/${id}`);
  },
};
