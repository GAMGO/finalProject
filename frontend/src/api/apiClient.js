// 기존에 만들어 둔 axios 인스턴스
import axios from "axios";

const apiClient = axios.create({
  baseURL: process.env.VITE_BASE_URL,
});

// 즐겨찾기 API
export const favoriteApi = {
  async getAll() {
    const res = await apiClient.get("/favorites");
    // 백엔드에서 ApiResponse<T> 쓰면 보통 { status, message, data } 구조일 거라서
    return res.data.data ?? res.data; // 둘 중 프로젝트 구조에 맞는 쪽으로 쓰이면 된다.
  },

  async create(favorite) {
    const res = await apiClient.post("/favorites", favorite);
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
  console.log("Access Token 저장 완료");
  //token.substring(0, 10) + "..."
};

//5. 토큰 관리 함수: 로그아웃 또는 토큰 만료 시 메모리 토큰을 제거합니다.
export const clearAuthToken = () => {
  globalAccessToken = null;
  console.log("Access Token 제거 완료.");
  // TODO: 실제 프로젝트에서는 여기에 로그인 페이지로 리다이렉트하는 로직을 추가합니다.
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

// 응답 인터셉터: 토큰 만료(401) 처리 로직
apiClient.interceptors.response.use(
  (response) => {
    // 2xx 응답은 그대로 반환
    return response;
  },
  (error) => {
    const originalRequest = error.config;

    // 서버에서 401 Unauthorized 에러를 보냈고, 재시도 플래그가 없는 경우 (무한 루프 방지)
    if (
      error.response &&
      error.response.status === 401 &&
      !originalRequest._retry
    ) {
      console.warn(
        "401 Unauthorized 감지. 토큰 만료로 간주하고 로그아웃 처리 시작."
      );

      // 1. 재시도 플래그 설정
      originalRequest._retry = true;

      // 2. 메모리 토큰 제거 및 리다이렉트 준비
      clearAuthToken();

      // 3. 사용자에게 알림 후 로그인 페이지로 리다이렉트 (실제 환경에서는 모달 사용 권장)
      setTimeout(() => {
        // alert 대신 프로젝트의 메시지 UI를 사용하세요.
        alert("인증 세션이 만료되었습니다. 다시 로그인해 주세요.");
        // 예: navigate('/login');
      }, 0);

      // 에러 전파를 막아 다음 .catch() 블록이 실행되지 않게 합니다.
      return Promise.reject(error);
    }

    // 401이 아닌 다른 에러는 그대로 전파
    return Promise.reject(error);
  }
);

export default apiClient;