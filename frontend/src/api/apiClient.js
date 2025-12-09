// src/api/apiClient.js
import axios from "axios";

// ===================================================
// 기본 axios 인스턴스 (모든 API 공용)
// ===================================================
const apiClient = axios.create({
  baseURL: import.meta.env.VITE_BASE_URL
});

// 공통 언랩 헬퍼
// - { status, message, data } 형태면 data 리턴
// - 아니면 res.data 그대로 리턴
const unwrap = (res) =>
  res.data && typeof res.data === "object" && "data" in res.data
    ? res.data.data
    : res.data;

// ===================================================
// 즐겨찾기 API
// ===================================================
export const favoriteApi = {
  // 전체 조회
  async getAll() {
    // 백엔드 매핑: GET /api/favorites  (FavoriteController 기준)
    const res = await apiClient.get("/api/favorites");
    return unwrap(res); // -> List<FavoriteResponse> 또는 ApiResponse<List<...>>
  },

  // 생성
  async create(favorite) {
    // POST /api/favorites
    const res = await apiClient.post("/api/favorites", favorite);
    return unwrap(res); // -> FavoriteResponse 또는 ApiResponse<FavoriteResponse>
  },

  // 수정
  async update(id, favorite) {
    // PUT /api/favorites/{id}
    const res = await apiClient.put(`/api/favorites/${id}`, favorite);
    return unwrap(res);
  },

  // 삭제
  async remove(id) {
    // DELETE /api/favorites/{id}
    await apiClient.delete(`/api/favorites/${id}`);
  },
};

// =================================================================
// JWT 인증 및 토큰 관리 로직
// =================================================================

// 🔑 전역 JWT 토큰 변수 (메모리 저장소 역할)
let globalAccessToken = null;
let globalRefreshToken = localStorage.getItem("refreshToken");

// 🔎 토큰 조회 함수 (메모리 → localStorage 순서로 확인)
const getTokenFromStorage = () => {
  if (globalAccessToken) return globalAccessToken;

  const stored = localStorage.getItem("jwtToken");
  if (stored) {
    globalAccessToken = stored;
  }
  return stored;
};

const getRefreshTokenFromStorage = () => {
  if (!globalRefreshToken) {
    globalRefreshToken = localStorage.getItem("refreshToken");
  }
  return globalRefreshToken;
};

// 토큰 세팅
export const setAuthToken = (token, refreshToken) => {
  const MIN_TOKEN_LENGTH = 50;

  // 1. Access Token 처리
  if (typeof token === "string" && token.length > MIN_TOKEN_LENGTH) {
    globalAccessToken = token;
    localStorage.setItem("jwtToken", token);
    console.log("✅ Access Token 설정 완료. 길이:", token.length);
  } else {
    globalAccessToken = null;
    localStorage.removeItem("jwtToken");
    if (token) {
      console.error(
        "❌ Access Token이 유효하지 않거나 너무 짧아 저장을 건너뛰고 기존 토큰을 제거했습니다."
      );
    }
  }

  // 2. Refresh Token 처리
  //    refreshToken 인자가 undefined/null 이면 기존 값 유지
  if (refreshToken === null || typeof refreshToken === "undefined") {
    console.log(
      "⚠️ Refresh Token 인자가 누락되어, 기존 저장소 값을 유지합니다."
    );
    return;
  }

  if (
    typeof refreshToken === "string" &&
    refreshToken.length > MIN_TOKEN_LENGTH
  ) {
    globalRefreshToken = refreshToken;
    localStorage.setItem("refreshToken", refreshToken);
    console.log("✅ Refresh Token 설정 완료. 길이:", refreshToken.length);
  } else {
    console.error(`❌ Refresh Token 제거됨! (인자 값: ${refreshToken})`);
    globalRefreshToken = null;
    localStorage.removeItem("refreshToken");
  }
};

// 토큰 삭제 (로그아웃 / 만료 시)
export const clearAuthToken = () => {
  globalAccessToken = null;
  globalRefreshToken = null;
  localStorage.removeItem("jwtToken");
  localStorage.removeItem("refreshToken");
  console.log("Access Token 제거 완료.");
  window.location.href = "/login";
};

// Refresh Token 으로 Access Token 재발급
const refreshAccessToken = async () => {
  const refreshToken = getRefreshTokenFromStorage();
  if (!refreshToken) {
    console.error("Refresh Token이 없습니다. 로그인 필요.");
    clearAuthToken();
    throw new Error("No Refresh Token");
  }

  try {
    // 기본 axios 사용 (apiClient 아님 → 인터셉터 루프 방지)
    const response = await axios.post(
      `${apiClient.defaults.baseURL}/api/auth/refresh`,
      {
        refreshToken: refreshToken,
      }
    );

    const newAccessToken = response.data.token;
    const newRefreshToken = response.data.refreshToken;

    if (newAccessToken) {
      setAuthToken(newAccessToken, newRefreshToken);
      console.log("Access Token 재발급 성공.");
      return newAccessToken;
    } else {
      clearAuthToken();
      throw new Error("Token refresh failed");
    }
  } catch (refreshError) {
    console.error(
      "Access Token 갱신 실패: Refresh Token도 만료되었을 수 있습니다.",
      refreshError
    );
    clearAuthToken();
    throw refreshError;
  }
};

// ===================================================
// 요청 인터셉터: 모든 요청에 Authorization 자동 주입
// ===================================================
apiClient.interceptors.request.use(
  (config) => {
    const currentToken = getTokenFromStorage();
    if (currentToken) {
      config.headers.Authorization = `Bearer ${currentToken}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

// ===================================================
// 응답 인터셉터: 401 → 토큰 리프레시 후 재요청
// ===================================================
apiClient.interceptors.response.use(
  (response) => response,
  async (error) => {
    const originalRequest = error.config;
    const status = error.response ? error.response.status : null;

    if (status === 401 && !originalRequest._retry) {
      originalRequest._retry = true;

      const refreshToken = getRefreshTokenFromStorage();
      if (!refreshToken) {
        console.warn("Refresh Token 없음. 로그인 페이지로 리다이렉트.");
        clearAuthToken();
        return Promise.reject(error);
      }

      try {
        const newAccessToken = await refreshAccessToken();
        originalRequest.headers.Authorization = `Bearer ${newAccessToken}`;
        console.log("만료된 요청을 새 Access Token으로 재시도 중...");
        return apiClient(originalRequest);
      } catch (refreshError) {
        console.error("Access Token 갱신 실패. 재 로그인 필요.");
        return Promise.reject(refreshError);
      }
    }

    return Promise.reject(error);
  }
);

export default apiClient;
