// src/main/java/org/iclass/finalproject/user/repository/LoginHistoryRepository.java
package org.iclass.finalproject.user.repository;

import org.iclass.finalproject.user.model.LoginHistory;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;

import java.sql.Date;
import java.util.List;

public interface LoginHistoryRepository extends JpaRepository<LoginHistory, Long> {

    @Query("select distinct lh.loginDate " +
           "from LoginHistory lh " +
           "where lh.user.id = :userId " +
           "order by lh.loginDate asc")
    List<Date> findLoginDates(Long userId);
}

/*
 * [파일 설명]
 * - 특정 유저가 로그인했던 날짜 목록을 오름차순으로 조회하는 레포지토리.
 * - 서비스에서 이 결과를 가지고 달력/총일수/연속일수 계산에 사용.
 */
