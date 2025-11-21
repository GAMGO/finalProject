// src/components/ChangePasswordForm.jsx
import React, { useState } from 'react';
import { api } from '../apiClient';

export function ChangePasswordForm() {
  const [currentPassword, setCurrentPassword] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [msg, setMsg] = useState('');
  const [err, setErr] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    setMsg('');
    setErr('');
    try {
      await api.changePassword(currentPassword, newPassword);
      setMsg('비밀번호가 변경되었습니다.');
      setCurrentPassword('');
      setNewPassword('');
    } catch (e) {
      setErr(e.message);
    }
  };

  return (
    <form onSubmit={handleSubmit} style={styles.form}>
      <h3>비밀번호 변경</h3>
      <label>
        현재 비밀번호
        <input
          type="password"
          value={currentPassword}
          onChange={(e) => setCurrentPassword(e.target.value)}
        />
      </label>
      <label>
        새 비밀번호
        <input
          type="password"
          value={newPassword}
          onChange={(e) => setNewPassword(e.target.value)}
        />
      </label>
      {err && <div style={styles.error}>{err}</div>}
      {msg && <div style={styles.success}>{msg}</div>}
      <button type="submit">변경</button>
    </form>
  );
}

const styles = {
  form: {
    display: 'flex',
    flexDirection: 'column',
    gap: 8,
    maxWidth: 300,
  },
  error: { color: 'red', fontSize: 12 },
  success: { color: 'green', fontSize: 12 },
};

/*
 * [파일 설명]
 * - /api/me/change-password 호출하는 폼.
 * - 현재 비밀번호 + 새 비밀번호 입력해서 변경.
 */
