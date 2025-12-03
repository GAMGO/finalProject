package org.iclass.polls.repository;

import org.springframework.data.jpa.repository.JpaRepository;
import org.iclass.polls.entity.*;

public interface PollOptionRepository extends JpaRepository<PollOption, Long> {
}
