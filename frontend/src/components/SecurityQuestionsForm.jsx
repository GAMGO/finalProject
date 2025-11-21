// src/components/SecurityQuestionsForm.jsx
import React, { useState } from 'react';
import { api } from '../apiClient';

const QUESTION_OPTIONS = [
  { value: 'BIRTH_COUNTRY', label: '당신이 태어난 나라' },
  { value: 'CHILDHOOD_CITY', label: '어렸을 때 살던 도시' },
  { value: 'PET_NAME', label: '반려동물 이름' },
];

export function SecurityQuestionsForm() {
  const [selectedQuestions, setSelectedQuestions] = useState([
    { question: 'BIRTH_COUNTRY', answer: '' },
  ]);
  const [msg, setMsg] = useState('');
  const [err, setErr] = useState('');

  const handleQuestionChange = (idx, field, value) => {
    setSelectedQuestions((prev) =>
      prev.map((q, i) => (i === idx ? { ...q, [field]: value } : q)),
    );
  };

  const addQuestion = () => {
    setSelectedQuestions((prev) => [
      ...prev,
      { question: 'BIRTH_COUNTRY', answer: '' },
    ]);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setMsg('');
    setErr('');
    try {
      await api.setSecurityQuestions(selectedQuestions);
      setMsg('보안 질문이 저장되었습니다.');
    } catch (e) {
      setErr(e.message);
    }
  };

  return (
    <form onSubmit={handleSubmit} style={styles.form}>
      <h3>보안 질문 설정</h3>
      {selectedQuestions.map((q, idx) => (
        <div key={idx} style={styles.row}>
          <select
            value={q.question}
            onChange={(e) =>
              handleQuestionChange(idx, 'question', e.target.value)
            }
          >
            {QUESTION_OPTIONS.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </select>
          <input
            type="text"
            placeholder="답변"
            value={q.answer}
            onChange={(e) =>
              handleQuestionChange(idx, 'answer', e.target.value)
            }
          />
        </div>
      ))}
      <button type="button" onClick={addQuestion}>
        질문 추가
      </button>

      {err && <div style={styles.error}>{err}</div>}
      {msg && <div style={styles.success}>{msg}</div>}

      <button type="submit">저장</button>
    </form>
  );
}

const styles = {
  form: {
    display: 'flex',
    flexDirection: 'column',
    gap: 8,
    maxWidth: 400,
  },
  row: {
    display: 'flex',
    gap: 8,
  },
  error: { color: 'red', fontSize: 12 },
  success: { color: 'green', fontSize: 12 },
};

/*
 * [파일 설명]
 * - /api/me/security-questions 호출하는 폼.
 * - 여러 개의 보안 질문/답변을 한 번에 등록할 수 있게 구현.
 */
