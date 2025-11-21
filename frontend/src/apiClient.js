// src/apiClient.js

const API_BASE = 'http://localhost:8080'; // 백엔드 주소에 맞게 수정

async function request(path, options = {}) {
  const res = await fetch(API_BASE + path, {
    credentials: 'include', // 세션 쿠키 포함
    headers: {
      'Content-Type': 'application/json',
      ...(options.headers || {}),
    },
    ...options,
  });

  const data = await res.json().catch(() => ({}));

  if (!res.ok || data.success === false) {
    const msg = data.message || '요청 실패';
    throw new Error(msg);
  }
  return data.data; // ApiResponse<T>의 data 필드만 리턴
}

export const api = {
  login: (email, password) =>
    request('/api/auth/login', {
      method: 'POST',
      body: JSON.stringify({ email, password }),
    }),

  logout: () =>
    request('/api/auth/logout', { method: 'POST' }),

  signup: (username, email, password) =>
    request('/api/auth/signup', {
      method: 'POST',
      body: JSON.stringify({ username, email, password }),
    }),

  resetPasswordByQuestion: (payload) =>
    request('/api/auth/reset-password/question', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),

  getMe: () => request('/api/me'),

  getLoginHistory: () => request('/api/me/login-history'),

  changePassword: (currentPassword, newPassword) =>
    request('/api/me/change-password', {
      method: 'POST',
      body: JSON.stringify({ currentPassword, newPassword }),
    }),

  setSecurityQuestions: (questions) =>
    request('/api/me/security-questions', {
      method: 'POST',
      body: JSON.stringify({ questions }),
    }),
};

/*
 * [파일 설명]
 * - 백엔드 스프링 API를 호출하는 공통 클라이언트.
 * - ApiResponse<T> 포맷을 가정하고 data만 리턴.
 * - 세션 기반 인증을 위해 모든 요청에 credentials: 'include' 설정.
 */
