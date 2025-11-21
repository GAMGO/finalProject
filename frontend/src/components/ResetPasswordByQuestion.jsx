// src/components/ResetPasswordByQuestion.jsx
import React, { useState } from 'react';
import { api } from '../apiClient';

const QUESTION_OPTIONS = [
  { value: 'BIRTH_COUNTRY', label: '당신이 태어난 나라' },
  { value: 'CHILDHOOD_CITY', label: '어렸을 때 살던 도시' },
  { value: 'PET_NAME', label: '반려동물 이름' },
];

export function ResetPasswordByQuestion() {
  const [email, setEmail] = useState('');
  const [question, setQuestion] = useState('BIRTH_COUNTRY');
  const [answer, setAnswer] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [msg, setMsg] = useState('');
  const [err, setErr] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    setErr('');
    setMsg('');
    try {
      await api.resetPasswordByQuestion({
        email,
        question,
        answer,
        newPassword,
      });
      setMsg('비밀번호가 변경되었습니다. 로그인하세요.');
    } catch (e) {
      setErr(e.message);
    }
  };

  return (
    <form onSubmit={handleSubmit} style={styles.form}>
      <h2>비밀번호 찾기 (보안 질문)</h2>
      <label>
        이메일
        <input
          type="email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
        />
      </label>

      <label>
        보안 질문
        <select
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
        >
          {QUESTION_OPTIONS.map((q) => (
            <option key={q.value} value={q.value}>
              {q.label}
            </option>
          ))}
        </select>
      </label>

      <label>
        답변
        <input
          type="text"
          value={answer}
          onChange={(e) => setAnswer(e.target.value)}
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

      <button type="submit">비밀번호 변경</button>
    </form>
  );
}

const styles = {
  form: {
    display: 'flex',
    flexDirection: 'column',
    gap: 8,
    width: 350,
  },
  error: {
    color: 'red',
    fontSize: 12,
  },
  success: {
    color: 'green',
    fontSize: 12,
  },
};

/*
 * [파일 설명]
 * - /api/auth/reset-password/question 연동 화면.
 * - 이메일 + 보안 질문 + 답 + 새 비밀번호 입력받아서 백엔드에 전달.
 */
