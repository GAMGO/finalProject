// src/api/apiClient.js
// 기존에 만들어 둔 axios 인스턴스
import axios from "axios";

const apiClient = axios.create({
  baseURL: import.meta.env.VITE_LOCAL_BASE_URL,
});

// ======================
// 즐겨찾기 API
// ======================
export const favoriteApi = {
  // 전체 조회
  async getAll() {
    // 백엔드 매핑에 맞게 /api/favorites 로 고정
    const res = await apiClient.get("/api/favorites");
    // 백엔드에서 ApiResponse<T> 쓰면 보통 { status, message, data } 구조일 거라서
    return res.data.data ?? res.data; // 둘 중 프로젝트 구조에 맞는 쪽으로 쓰이면 된다.
  },

  // 생성
  async create(favorite) {
    const res = await apiClient.post("/api/favorites", favorite);
    return res.data.data ?? res.data;
  },

  // 수정
  async update(id, favorite) {
    const res = await apiClient.put(`/api/favorites/${id}`, favorite);
    return res.data.data ?? res.data;
  },

  // 삭제
  async remove(id) {
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

export const setAuthToken = (token, refreshToken) => {
  const MIN_TOKEN_LENGTH = 50;
  // 1. Access Token 처리 (항상 저장 또는 제거)
  if (typeof token === "string" && token.length > MIN_TOKEN_LENGTH) {
    globalAccessToken = token;
    localStorage.setItem("jwtToken", token);
    console.log("✅ Access Token 설정 완료. 길이:", token.length);
  } else {
    globalAccessToken = null;
    localStorage.removeItem("jwtToken"); // LocalStorage에서 제거
    if (token) {
      console.error(
        "❌ Access Token이 유효하지 않거나 너무 짧아 저장을 건너뛰고 기존 토큰을 제거했습니다."
      );
    }
  } // 2. Refresh Token 처리 // 🚨 [핵심 수정] refreshToken 인자가 undefined나 null이면 기존 값 유지 (제거하지 않음)
  if (refreshToken === null || typeof refreshToken === "undefined") {
    console.log(
      "⚠️ Refresh Token 인자가 누락되어, 기존 저장소 값을 유지합니다."
    );
    return; // Access Token만 처리하고 종료
  } // 인자가 유효한 토큰인 경우 (갱신 또는 새로 저장)

  if (
    typeof refreshToken === "string" &&
    refreshToken.length > MIN_TOKEN_LENGTH
  ) {
    globalRefreshToken = refreshToken; // Refresh Token은 localStorage에 저장
    localStorage.setItem("refreshToken", refreshToken);
    console.log("✅ Refresh Token 설정 완료. 길이:", refreshToken.length);
  } else {
    // 인자가 유효하지 않으므로 (빈 문자열 등), 강제로 제거
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

// Refresh Token 요청 함수 (내부 사용)
const refreshAccessToken = async () => {
  const refreshToken = getRefreshTokenFromStorage();
  if (!refreshToken) {
    console.error("Refresh Token이 없습니다. 로그인 필요.");
    clearAuthToken();
    throw new Error("No Refresh Token");
  }
  try {
    // ⭐️ 기본 axios를 사용하여 토큰 재발급 요청 (무한 루프 방지)
    // 백엔드 구현에 따라 Refresh Token을 Header나 Body에 담아 요청합니다.
    const response = await axios.post(
      `${apiClient.defaults.baseURL}/api/auth/refresh`,
      {
        refreshToken: refreshToken,
      }
    );
    const newAccessToken = response.data.token;
    const newRefreshToken = response.data.refreshToken; // 백엔드에서 새로운 리프레시 토큰도 주는 경우
    if (newAccessToken) {
      // ⭐️ 새로운 Access Token 및 Refresh Token 저장
      setAuthToken(newAccessToken, newRefreshToken);
      console.log("Access Token 재발급 성공.");
      return newAccessToken;
    } else {
      // 서버에서 토큰을 주지 않은 경우 (Refresh Token도 만료되었을 가능성)
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
// 6. 요청 인터셉터 설정 (모든 요청에 토큰 자동 주입)
apiClient.interceptors.request.use(
  (config) => {
    const currentToken = getTokenFromStorage();
    if (currentToken) {
      // 모든 요청에 'Authorization: Bearer <토큰>' 헤더를 자동으로 추가
      config.headers.Authorization = `Bearer ${currentToken}`;
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
  // ⭐️ [수정] 401 에러 발생 시 Access Token 갱신 로직 추가
  async (error) => {
    const originalRequest = error.config;
    const status = error.response ? error.response.status : null;

    // 1. 서버에서 401 Unauthorized 에러를 보냈고, 재시도 플래그가 없는 경우 (무한 루프 방지)
    if (status === 401 && !originalRequest._retry) {
      originalRequest._retry = true; // 재시도 플래그 설정 (무한 루프 방지)

      // 2. Refresh Token이 있는지 확인
      const refreshToken = getRefreshTokenFromStorage();
      if (!refreshToken) {
        console.warn("Refresh Token 없음. 로그인 페이지로 리다이렉트.");
        clearAuthToken();
        return Promise.reject(error);
      }

      try {
        // 3. Access Token 갱신 시도
        const newAccessToken = await refreshAccessToken();

        // 4. 원래 요청의 Authorization 헤더를 새 토큰으로 업데이트
        originalRequest.headers.Authorization = `Bearer ${newAccessToken}`;

        // 5. 원래 요청 재시도
        console.log("만료된 요청을 새 Access Token으로 재시도 중...");
        return apiClient(originalRequest);
      } catch (refreshError) {
        // 갱신 실패 시 (예: Refresh Token도 만료) -> 로그아웃 처리
        console.error("Access Token 갱신 실패. 재 로그인 필요.");
        // refreshAccessToken 내부에서 이미 clearAuthToken을 호출합니다.
        return Promise.reject(refreshError);
      }
    }

    // 401 에러가 아니거나, 이미 재시도한 요청이거나, 요청 자체가 실패한 경우
    return Promise.reject(error);
  }
);
// 기존 export 구문 유지
export default apiClient;
