package org.iclass.finalproject.user.dto;

import lombok.AllArgsConstructor;
import lombok.Data;

import java.time.LocalDate;
import java.util.List;

@Data
@AllArgsConstructor
public class LoginHistoryResponse {

    // 로그인했던 날짜 리스트 (중복 없는 날짜)
    private List<LocalDate> dates;

    // 총 로그인한 일수 (distinct 날짜 개수)
    private int totalDays;

    // 현재 연속 로그인 일수 (마지막 로그인 날짜 기준으로 역산)
    private int currentStreak;
}

/*
 * [파일 설명]
 * - 로그인 기록 API 응답 DTO.
 * - dates: 달력에 동그라미 칠 날짜들.
 * - totalDays: 지금까지 로그인한 날짜 수.
 * - currentStreak: 끊기지 않고 연속으로 로그인한 일수(출석 연속일).
 */
