// src/api/comments.js
import apiClient from "./apiClient";

// 댓글 목록 (Page<CommentResponse>)
export async function listComments(postId, page = 0, size = 10) {
  const res = await apiClient.get(`/api/posts/${postId}/comments`, {
    params: { page, size },
  });
  return res.data; // { content, number, totalPages, ... }
}

// 댓글 작성
export async function createComment(postId, { author, content }) {
  const res = await apiClient.post(`/api/posts/${postId}/comments`, {
    author,
    content,
  });
  return res.data; // 보통 생성된 commentId
}

// 댓글 삭제
export async function deleteComment(commentId) {
  await apiClient.delete(`/api/comments/${commentId}`);
}
