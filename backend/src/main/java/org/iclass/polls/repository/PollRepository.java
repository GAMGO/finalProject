package org.iclass.polls.repository;

import org.springframework.data.jpa.repository.JpaRepository;
import org.iclass.polls.entity.*;

public interface PollRepository extends JpaRepository<Poll, Long> {
}
