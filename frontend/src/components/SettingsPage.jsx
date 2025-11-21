// src/components/SettingsPage.jsx
import React, { useEffect, useState } from 'react';
import { api } from '../apiClient';
import { LoginHistoryCalendar } from './LoginHistoryCalendar';
import { ChangePasswordForm } from './ChangePasswordForm';
import { SecurityQuestionsForm } from './SecurityQuestionsForm';

export function SettingsPage({ onLogout }) {
  const [profile, setProfile] = useState(null);
  const [loginHistory, setLoginHistory] = useState(null);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState('');

  useEffect(() => {
    const fetchData = async () => {
      try {
        const me = await api.getMe();
        const history = await api.getLoginHistory();
        setProfile(me);
        setLoginHistory(history);
      } catch (e) {
        setErr(e.message);
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, []);

  const handleLogout = async () => {
    await api.logout();
    onLogout && onLogout();
  };

  if (loading) return <div>로딩 중...</div>;
  if (err) return <div>에러: {err}</div>;
  if (!profile) return <div>정보 없음</div>;

  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <h1>설정</h1>
        <button onClick={handleLogout}>로그아웃</button>
      </div>

      <div style={styles.section}>
        <h2>회원 정보</h2>
        <p>사용자 이름: {profile.username}</p>
        <p>이메일: {profile.email}</p>
        <p>생일: {profile.birthday || '-'}</p>
        <p>성별: {profile.gender || '-'}</p>
        <p>주소: {profile.address || '-'}</p>
        <p>소개: {profile.bio || '-'}</p>
      </div>

      <div style={styles.flexRow}>
        <div style={styles.section}>
          <h2>로그인 기록</h2>
          {loginHistory && (
            <>
              <p>총 로그인 일수: {loginHistory.totalDays}</p>
              <p>연속 로그인 일수: {loginHistory.currentStreak}</p>
              <LoginHistoryCalendar dates={loginHistory.dates} />
            </>
          )}
        </div>

        <div style={styles.section}>
          <ChangePasswordForm />
          <SecurityQuestionsForm />
        </div>
      </div>
    </div>
  );
}

const styles = {
  container: {
    padding: 16,
    display: 'flex',
    flexDirection: 'column',
    gap: 16,
  },
  header: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  section: {
    border: '1px solid #ddd',
    borderRadius: 8,
    padding: 12,
    marginBottom: 12,
  },
  flexRow: {
    display: 'flex',
    gap: 16,
    alignItems: 'flex-start',
    flexWrap: 'wrap',
  },
};

/*
 * [파일 설명]
 * - 설정(마이페이지) 화면 전체를 구성하는 컴포넌트.
 * - /api/me, /api/me/login-history를 호출해 회원 정보 + 로그인 기록 표시.
 * - 로그인 기록은 LoginHistoryCalendar로 달력 + 총/연속 로그인 일수 표시.
 * - 비밀번호 변경, 보안질문 설정 폼도 같이 배치.
 */
