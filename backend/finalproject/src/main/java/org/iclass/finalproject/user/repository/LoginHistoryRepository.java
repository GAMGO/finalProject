package org.iclass.finalproject.user.repository;

import org.iclass.finalproject.user.model.LoginHistory;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;

import java.sql.Date;
import java.util.List;

public interface LoginHistoryRepository extends JpaRepository<LoginHistory, Long> {

    @Query(value = """
        SELECT DATE(login_at) AS login_date
        FROM login_history
        WHERE user_id = :userId
        GROUP BY login_date
        ORDER BY login_date
        """, nativeQuery = true)
    List<Date> findLoginDates(@Param("userId") Long userId);
}

/*
 * [파일 설명]
 * - LoginHistory 엔티티에 대한 Repository.
 * - 특정 회원의 로그인 "날짜" 목록을 한 번에 가져오는 쿼리를 제공.
 * - 출석 달력/연속 로그인 계산에서 사용되는 DB 접근 계층.
 */
