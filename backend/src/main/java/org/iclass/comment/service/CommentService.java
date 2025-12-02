package org.iclass.comment.service;

import jakarta.persistence.EntityManager;
import jakarta.persistence.EntityNotFoundException;
import lombok.RequiredArgsConstructor;
import org.iclass.comment.dto.CommentRequest;
import org.iclass.comment.dto.CommentResponse;
import org.iclass.community.Post;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.iclass.comment.entity.Comment;
import org.iclass.comment.repository.*;

@Service
@RequiredArgsConstructor
@Transactional(readOnly = true)
public class CommentService {

    private final CommentRepository commentRepository;
    private final EntityManager em; // PostRepository 없이 참조용으로 사용

    public Page<CommentResponse> listByPost(Long postId, Pageable pageable) {
        Post ref = em.getReference(Post.class, postId);
        Page<Comment> page = commentRepository.findByPostOrderByCreatedAtDesc(ref, pageable);
        return page.map(this::toResponse);
    }

    @Transactional
    public Long add(Long postId, CommentRequest req) {
        Post post = em.find(Post.class, postId);
        if (post == null) throw new EntityNotFoundException("게시글을 찾을 수 없습니다. id=" + postId);

        Comment c = new Comment();
        c.setPost(post);
        c.setAuthor(req.getAuthor());
        c.setContent(req.getContent());

        return commentRepository.save(c).getId();
    }

    @Transactional
    public void delete(Long commentId) {
        if (!commentRepository.existsById(commentId)) {
            throw new EntityNotFoundException("댓글을 찾을 수 없습니다. id=" + commentId);
        }
        commentRepository.deleteById(commentId);
    }

    private CommentResponse toResponse(Comment c) {
        CommentResponse dto = new CommentResponse();
        dto.setId(c.getId());
        dto.setPostIdx(c.getPost() != null ? c.getPost().getIdx() : null);
        dto.setAuthor(c.getAuthor());
        dto.setContent(c.getContent());
        dto.setCreatedAt(c.getCreatedAt());
        dto.setUpdatedAt(c.getUpdatedAt());
        return dto;
    }
}
