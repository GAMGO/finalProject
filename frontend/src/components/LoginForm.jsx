// src/components/LoginForm.jsx
import React, { useState } from 'react';
import { api } from '../apiClient';

export function LoginForm({ onLoggedIn }) {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [err, setErr] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setErr('');
    setLoading(true);
    try {
      await api.login(email, password);
      onLoggedIn && onLoggedIn();
    } catch (e) {
      setErr(e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <form onSubmit={handleSubmit} style={styles.form}>
      <h2>로그인</h2>
      <label>
        이메일
        <input
          type="email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
        />
      </label>
      <label>
        비밀번호
        <input
          type="password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
        />
      </label>
      {err && <div style={styles.error}>{err}</div>}
      <button type="submit" disabled={loading}>
        {loading ? '로그인 중...' : '로그인'}
      </button>
    </form>
  );
}

const styles = {
  form: {
    display: 'flex',
    flexDirection: 'column',
    gap: 8,
    width: 300,
  },
  error: {
    color: 'red',
    fontSize: 12,
  },
};

/*
 * [파일 설명]
 * - /api/auth/login 연동하는 로그인 폼 컴포넌트.
 * - 로그인 성공 시 onLoggedIn 콜백 호출 (상위에서 SettingsPage로 라우팅 등).
 */
