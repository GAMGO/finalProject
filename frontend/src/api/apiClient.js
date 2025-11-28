// src/api/apiClient.js
import axios from "axios";

// 팀 프로젝트 백엔드 : 8080
// baseURL 을 /api 까지 잡고, 나머지는 전부 /favorites, /profile 이런 식으로만 호출
const apiClient = axios.create({
  baseURL: "http://localhost:8080/api",
  timeout: 5000,
});

// ================= JWT 자동 실어 보내기 =================
apiClient.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem("jwtToken");
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

// ApiResponse 래핑 여부 상관없이 data 뽑는 도우미
const unwrap = (res) => {
  const data = res.data;
  if (data && typeof data === "object" && "data" in data) {
    return data.data;
  }
  return data;
};

// ================= 프로필 API =================
export async function fetchUserProfile() {
  // GET http://localhost:8080/api/profile
  const res = await apiClient.get("/profile");
  return unwrap(res);
}

export async function saveUserProfile(profile) {
  // POST http://localhost:8080/api/profile
  const res = await apiClient.post("/profile", profile);
  return unwrap(res);
}

// ================= 즐겨찾기 API =================
export const favoriteApi = {
  async getAll() {
    // GET http://localhost:8080/api/favorites
    const res = await apiClient.get("/favorites");
    return unwrap(res) ?? [];
  },

  async create(favorite) {
    // POST http://localhost:8080/api/favorites
    const res = await apiClient.post("/favorites", favorite);
    return unwrap(res);
  },

  async update(id, favorite) {
    // PUT http://localhost:8080/api/favorites/{id}
    const res = await apiClient.put(`/favorites/${id}`, favorite);
    return unwrap(res);
  },

  async remove(id) {
    // DELETE http://localhost:8080/api/favorites/{id}
    await apiClient.delete(`/favorites/${id}`);
  },
};

export default apiClient;
