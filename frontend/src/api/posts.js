// src/api/posts.js
import apiClient from "./apiClient";

/** 게시글 목록 (페이지네이션/검색은 백엔드 스펙에 맞춰 확장) */
export async function listPosts(params = {}) {
  const res = await apiClient.get("/api/posts", { params });
  return res.data; // Page<PostResponse> 또는 배열(백엔드 스펙에 맞춤)
}

/** 게시글 단건 조회 */
export async function getPost(postId) {
  const res = await apiClient.get(`/api/posts/${postId}`);
  return res.data;
}

/** 게시글 생성 (JSON 기반, 파일은 별도 업로드 시 imageUrl만 전달) */
export async function createPost(body) {
  const res = await apiClient.post("/api/posts", body);
  return res.data; // 새 postId 또는 생성 객체
}

/** 게시글 수정 (PUT or PATCH는 백엔드 스펙에 맞춰 사용) */
export async function updatePost(postId, body) {
  const res = await apiClient.put(`/api/posts/${postId}`, body);
  return res.data;
}

/** 게시글 삭제 */
export async function deletePost(postId) {
  await apiClient.delete(`/api/posts/${postId}`);
}
