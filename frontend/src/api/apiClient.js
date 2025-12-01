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
// =================================================================
// JWT 인증 및 토큰 관리 로직
// =================================================================
// 3. 🔑 전역 JWT 토큰 변수 (메모리 저장소 역할)
let globalAccessToken = null;

export const setAuthToken = (token) => {
    // ⭐️ 메모리 저장소에 토큰 저장 (XSS 공격으로부터 localStorage보다 안전)
    globalAccessToken = token; 
};

//5. 토큰 관리 함수: 로그아웃 또는 토큰 만료 시 메모리 토큰을 제거합니다.
export const clearAuthToken = () => {
    globalAccessToken = null;
};
// 6. 요청 인터셉터 설정 (모든 요청에 토큰 자동 주입)
apiClient.interceptors.request.use(
    (config) => {
        // 전역 토큰이 존재할 경우에만 헤더 추가
        if (globalAccessToken) {
            // ⭐️ 모든 요청에 'Authorization: Bearer <토큰>' 헤더를 자동으로 추가
            config.headers.Authorization = `Bearer ${globalAccessToken}`;
        }
        return config;
    },
    (error) => {
        return Promise.reject(error);
    }
);
export default apiClient;