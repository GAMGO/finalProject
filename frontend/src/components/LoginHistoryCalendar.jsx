// src/components/LoginHistoryCalendar.jsx
import React, { useMemo, useState } from 'react';

export function LoginHistoryCalendar({ dates }) {
  // dates: ["2025-11-20", ...] 형태라 가정
  const [monthOffset, setMonthOffset] = useState(0);

  const today = new Date();
  const baseYear = today.getFullYear();
  const baseMonth = today.getMonth(); // 0~11

  const targetDate = new Date(baseYear, baseMonth + monthOffset, 1);
  const year = targetDate.getFullYear();
  const month = targetDate.getMonth(); // 0~11

  const loginDateSet = useMemo(() => {
    return new Set(dates); // "YYYY-MM-DD"
  }, [dates]);

  const firstDayOfWeek = new Date(year, month, 1).getDay(); // 0(일)~6(토)
  const daysInMonth = new Date(year, month + 1, 0).getDate();

  const cells = [];
  for (let i = 0; i < firstDayOfWeek; i++) {
    cells.push(null);
  }
  for (let d = 1; d <= daysInMonth; d++) {
    cells.push(d);
  }

  const formatDate = (y, m, d) =>
    `${y}-${String(m + 1).padStart(2, '0')}-${String(d).padStart(2, '0')}`;

  return (
    <div style={styles.wrapper}>
      <div style={styles.header}>
        <button onClick={() => setMonthOffset((prev) => prev - 1)}>{'<'}</button>
        <span>
          {year}년 {month + 1}월
        </span>
        <button onClick={() => setMonthOffset((prev) => prev + 1)}>{'>'}</button>
      </div>
      <div style={styles.grid}>
        {['일', '월', '화', '수', '목', '금', '토'].map((w) => (
          <div key={w} style={styles.weekday}>
            {w}
          </div>
        ))}
        {cells.map((day, idx) => {
          if (day === null) {
            return <div key={idx} style={styles.cell} />;
          }
          const key = formatDate(year, month, day);
          const loggedIn = loginDateSet.has(key);
          return (
            <div key={idx} style={styles.cell}>
              <div
                style={{
                  ...styles.dayCircle,
                  ...(loggedIn ? styles.dayCircleLogged : {}),
                }}
              >
                {day}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

const styles = {
  wrapper: {
    border: '1px solid #ccc',
    borderRadius: 8,
    padding: 12,
    maxWidth: 350,
  },
  header: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  grid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(7, 1fr)',
    gap: 4,
  },
  weekday: {
    textAlign: 'center',
    fontWeight: 'bold',
    fontSize: 12,
  },
  cell: {
    height: 32,
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
  },
  dayCircle: {
    width: 24,
    height: 24,
    borderRadius: '50%',
    fontSize: 12,
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
  },
  dayCircleLogged: {
    backgroundColor: '#ff4d4f',
    color: '#fff',
  },
};

/*
 * [파일 설명]
 * - 로그인 날짜 리스트를 달력으로 렌더링.
 * - 로그인한 날은 빨간 동그라미로 표시해서 "출석" 느낌 구현.
 * - monthOffset으로 이전/다음 달 이동 가능.
 */
