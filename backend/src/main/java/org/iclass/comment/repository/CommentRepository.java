package org.iclass.comment.repository;

import org.iclass.community.Post;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;
import org.iclass.comment.entity.Comment;

public interface CommentRepository extends JpaRepository<Comment, Long> {

    Page<Comment> findByPostOrderByCreatedAtDesc(Post post, Pageable pageable);
}
