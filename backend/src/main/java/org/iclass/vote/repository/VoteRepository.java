package org.iclass.vote.repository;

import org.springframework.data.jpa.repository.JpaRepository;
import org.iclass.vote.entity.*;

public interface VoteRepository extends JpaRepository<Vote, Long> {
}
