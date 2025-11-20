package org.iclass.finalproject.user.service;

import lombok.RequiredArgsConstructor;
import org.iclass.finalproject.user.dto.LoginHistoryResponse;
import org.iclass.finalproject.user.model.LoginHistory;
import org.iclass.finalproject.user.model.User;
import org.iclass.finalproject.user.repository.LoginHistoryRepository;
import org.iclass.finalproject.user.repository.UserRepository;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.sql.Date;
import java.time.LocalDate;
import java.util.List;

@Service
@RequiredArgsConstructor
public class LoginHistoryService {

    private final LoginHistoryRepository loginHistoryRepository;
    private final UserRepository userRepository;

    @Transactional
    public void recordLogin(Long userId) {
        User u = userRepository.findById(userId)
                .orElseThrow(() -> new IllegalArgumentException("user not found"));
        LoginHistory history = LoginHistory.of(u);
        loginHistoryRepository.save(history);
    }

    @Transactional(readOnly = true)
    public LoginHistoryResponse getHistory(Long userId) {
        List<LocalDate> dates = loginHistoryRepository.findLoginDates(userId).stream()
                .map(Date::toLocalDate)
                .toList();

        int totalDays = dates.size();
        int currentStreak = calculateCurrentStreak(dates);

        return new LoginHistoryResponse(dates, totalDays, currentStreak);
    }

    private int calculateCurrentStreak(List<LocalDate> dates) {
        if (dates.isEmpty()) return 0;

        // dates는 쿼리에서 ORDER BY login_date로 정렬돼 있는 상태 (오름차순)
        int streak = 1;

        for (int i = dates.size() - 1; i > 0; i--) {
            LocalDate today = dates.get(i);
            LocalDate prev = dates.get(i - 1);

            if (prev.plusDays(1).equals(today)) {
                streak++;
            } else {
                // 끊기는 순간 멈춘다 (현재 연속일만 본다)
                break;
            }
        }

        return streak;
    }
}

/*
 * [파일 설명]
 * - 로그인 기록을 저장하고, 로그인 통계를 계산하는 서비스.
 * - recordLogin(userId): 로그인 성공 시 호출, DB에 1줄 insert.
 * - getHistory(userId): 날짜 목록 + 총 로그인 일수 + 현재 연속 로그인 일수 계산 후 반환.
 */
