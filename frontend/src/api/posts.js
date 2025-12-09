// src/api/posts.js
import apiClient from "./apiClient";

/** 게시글 목록 조회 */
export async function listPosts(params = {}) {
  const res = await apiClient.get("/api/posts", { params });
  return res.data; // Page<PostResponse> 또는 배열
}

/** 게시글 단건 조회 */
export async function getPost(postId) {
  const res = await apiClient.get(`/api/posts/${postId}`);
  return res.data;
}

/** 게시글 생성 */
export async function createPost(form) {
  const payload = {
    title: form.title,
    body: form.body,
    writer: form.writer,
    locationText: form.locationText,
    storeCategory: form.storeCategory,
    type: form.type,
    imageUrl: form.imageUrl,
  };

  const res = await apiClient.post("/api/posts", payload);
  return res.data;
}

/** 게시글 수정 */
export async function updatePost(postId, form) {
  const payload = {
    title: form.title,
    body: form.body,
    writer: form.writer,
    locationText: form.locationText,
    storeCategory: form.storeCategory,
    type: form.type,
    imageUrl: form.imageUrl,
  };

  const res = await apiClient.put(`/api/posts/${postId}`, payload);
  return res.data;
}

/** 게시글 삭제 */
export async function deletePost(postId) {
  await apiClient.delete(`/api/posts/${postId}`);
}
