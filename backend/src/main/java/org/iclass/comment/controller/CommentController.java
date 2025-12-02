package org.iclass.comment.controller;

import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.iclass.comment.dto.CommentRequest;
import org.iclass.comment.dto.CommentResponse;
import org.iclass.comment.service.CommentService;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageRequest;
import org.springframework.web.bind.annotation.*;

@RestController
@RequiredArgsConstructor
@RequestMapping("/api")
public class CommentController {

    private final CommentService commentService;

    // 댓글 목록 (페이징)
    @GetMapping("/posts/{postId}/comments")
    public Page<CommentResponse> list(
            @PathVariable Long postId,
            @RequestParam(defaultValue = "0") int page,
            @RequestParam(defaultValue = "10") int size
    ) {
        return commentService.listByPost(postId, PageRequest.of(page, size));
    }

    // 댓글 작성
    @PostMapping("/posts/{postId}/comments")
    public Long create(@PathVariable Long postId, @Valid @RequestBody CommentRequest req) {
        return commentService.add(postId, req);
    }

    // 댓글 삭제
    @DeleteMapping("/comments/{commentId}")
    public void delete(@PathVariable Long commentId) {
        commentService.delete(commentId);
    }
}
