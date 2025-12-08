// src/pages/CommunityPage.jsx
import React, { useEffect, useState } from "react";
import "../components/community/Community.css";
import PostEditor from "../components/community/PostEditor";
import writeFabIcon from "../assets/write-icon.svg";

// API
import {
  listPosts as listPostsApi,
  createPost as createPostApi,
  updatePost as updatePostApi,
  deletePost as deletePostApi,
} from "../api/posts";
import {
  listComments,
  createComment,
  deleteComment,
} from "../api/comments";

const DEFAULT_THUMB =
  "https://images.pexels.com/photos/461198/pexels-photo-461198.jpeg?auto=compress&cs=tinysrgb&w=400";

/** 서버(PostResponse) → 커뮤니티 카드용 포맷으로 매핑 */
function mapPostFromApi(p) {
  return {
    // ⚠️ DB 컬럼이 idx여도 프론트에서는 id로 통일
    id: p.id ?? p.postId ?? p.idx,
    title: p.title ?? "(제목 없음)",
    writer: p.writer ?? "익명",
    board: p.storeCategory ?? p.board ?? "노점",
    time: p.createdAt
      ? new Date(p.createdAt).toLocaleTimeString("ko-KR", {
          hour: "2-digit",
          minute: "2-digit",
        })
      : (p.created_at
          ? new Date(p.created_at).toLocaleTimeString("ko-KR", {
              hour: "2-digit",
              minute: "2-digit",
            })
          : "방금 전"),
    commentCount: p.commentCount ?? 0,
    views: p.viewCount ?? p.views ?? 0,
    category: p.type ?? p.category ?? "제보",
    location: p.locationText ?? p.location ?? "",
    // 빈 문자열("")도 기본 썸네일로 대체
    thumbnail: p.imageUrl || DEFAULT_THUMB,
    // 게시글 본문은 body로 통일
    body: p.body ?? p.content ?? "",
  };
}

/** 커뮤니티 카드 → PostEditor 초기값 포맷 */
function mapPostToEditorInitial(post) {
  return {
    type: post.category ?? "제보",
    title: post.title ?? "",
    body: post.body ?? "",                  // content 대신 body
    locationText: post.location ?? "",
    storeCategory: post.board ?? "",
    writer: post.writer ?? "",
    imageUrl: post.thumbnail ?? "",
  };
}

export default function CommunityPage() {
  // view: list / detail / write
  const [view, setView] = useState("list");
  const [posts, setPosts] = useState([]);
  const [selectedPostId, setSelectedPostId] = useState(null);
  const [editingPost, setEditingPost] = useState(null); // PostEditor 초기값(수정 모드일 때만 세팅)

  // 댓글 상태: { [postId]: [{ id, author, content, createdAt }] }
  const [commentsByPost, setCommentsByPost] = useState({});

  const selectedPost =
    posts.find((p) => p.id === selectedPostId) || null;

  // ─────────────────────────────────────────────────────────────
  // 게시글 목록 최초 로드
  // ─────────────────────────────────────────────────────────────
  useEffect(() => {
    (async () => {
      try {
        const data = await listPostsApi();
        const list = Array.isArray(data) ? data : data?.content ?? [];
        setPosts(list.map(mapPostFromApi));
      } catch (e) {
        console.error("[listPosts] 실패:", e);
      }
    })();
  }, []);

  // ─────────────────────────────────────────────────────────────
  // 상세 진입: 댓글 불러오기(페이지 0 기준)
  // ─────────────────────────────────────────────────────────────
  const handleOpenPost = async (postId) => {
    if (!postId) {
      console.warn("postId가 없습니다.");
      alert("게시글 정보를 불러올 수 없어요.");
      return;
    }
    setSelectedPostId(postId);
    setView("detail");
    try {
      const page = await listComments(postId, 0, 10);
      setCommentsByPost((prev) => ({
        ...prev,
        [postId]: page?.content ?? [],
      }));
    } catch (e) {
      console.error("[listComments] 실패:", e);
      alert("댓글을 불러오지 못했어요.");
    }
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  const handleBackToList = () => {
    setSelectedPostId(null);
    setView("list");
  };

  // ─────────────────────────────────────────────────────────────
  // 댓글 등록
  // ─────────────────────────────────────────────────────────────
  const handleAddComment = async (postId, content) => {
    if (!content.trim()) return;
    try {
      await createComment(postId, {
        author: "익명",
        content: content.trim(),
      });
      const page = await listComments(postId, 0, 10);
      setCommentsByPost((prev) => ({
        ...prev,
        [postId]: page?.content ?? [],
      }));
    } catch (e) {
      console.error("[createComment] 실패:", e);
      alert("댓글 등록에 실패했어요.");
    }
  };

  // ─────────────────────────────────────────────────────────────
  // 글쓰기(생성) / 수정 / 삭제
  // ─────────────────────────────────────────────────────────────
  const handleOpenWrite = () => {
    setEditingPost(null); // 생성 모드
    setView("write");
  };

  const handleCreatePost = async (form) => {
    try {
      const created = await createPostApi(form); // 서버가 id 또는 객체 반환
      // 목록 재조회
      const data = await listPostsApi();
      const list = Array.isArray(data) ? data : data?.content ?? [];
      const mapped = list.map(mapPostFromApi);
      setPosts(mapped);

      // 생성된 글로 이동 (id / postId / idx 다 대응)
      const createdId =
        (typeof created === "object"
          ? created.id ?? created.postId ?? created.idx
          : created) ?? mapped[0]?.id;
      setSelectedPostId(createdId ?? null);
      setView("detail");
    } catch (e) {
      console.error("[createPost] 실패:", e);
      alert("게시글 등록에 실패했어요.");
    }
  };

  const openEdit = (post) => {
    setEditingPost(mapPostToEditorInitial(post)); // 수정 모드 초기값
    setSelectedPostId(post.id);
    setView("write");
  };

  const handleUpdatePost = async (postId, form) => {
    try {
      await updatePostApi(postId, form);
      const data = await listPostsApi();
      const list = Array.isArray(data) ? data : data?.content ?? [];
      const mapped = list.map(mapPostFromApi);
      setPosts(mapped);
      setEditingPost(null);
      setSelectedPostId(postId);
      setView("detail");
    } catch (e) {
      console.error("[updatePost] 실패:", e);
      alert("게시글 수정에 실패했어요.");
    }
  };

  const handleDeletePost = async (postId) => {
    if (!confirm("이 게시글을 삭제할까요?")) return;
    try {
      await deletePostApi(postId);
      const data = await listPostsApi();
      const list = Array.isArray(data) ? data : data?.content ?? [];
      setPosts(list.map(mapPostFromApi));
      setSelectedPostId(null);
      setView("list");
    } catch (e) {
      console.error("[deletePost] 실패:", e);
      alert("게시글 삭제에 실패했어요.");
    }
  };

  const handleCloseEditor = () => {
    setEditingPost(null);
    setView("list");
  };

  return (
    <div className="community-root">
      {/* 리스트 화면 */}
      {view === "list" && (
        <>
          <div className="community-input-wrapper">
            <input
              className="community-main-input"
              type="text"
              placeholder="노점 제보, 후기, 장소를 입력해 보세요"
            />
          </div>

          <CommunityList posts={posts} onOpenPost={handleOpenPost} />

          {/* 오른쪽 하단 글쓰기 버튼 */}
          <button
            type="button"
            className="community-fab"
            onClick={handleOpenWrite}
          >
            <img src={writeFabIcon} alt="글쓰기" />
          </button>
        </>
      )}

      {/* 상세 화면 */}
      {view === "detail" && selectedPost && (
        <CommunityDetail
          post={selectedPost}
          allPosts={posts}
          comments={commentsByPost[selectedPost.id] || []}
          onBack={handleBackToList}
          onAddComment={handleAddComment}
          onEdit={() => openEdit(selectedPost)}
          onDelete={() => handleDeletePost(selectedPost.id)}
          onDeleteComment={(commentId) => deleteComment(commentId)} // 필요 시 사용
        />
      )}

      {/* 글쓰기/수정 화면 (PostEditor 재사용) */}
      {view === "write" && (
        <PostEditor
          initial={editingPost ?? undefined}
          onClose={handleCloseEditor}
          onSubmit={async (form) => {
            if (editingPost && selectedPostId) {
              await handleUpdatePost(selectedPostId, form);
            } else {
              await handleCreatePost(form);
            }
          }}
        />
      )}
    </div>
  );
}

/* ===========================
 *   리스트 컴포넌트
 * =========================== */

function CommunityList({ posts, onOpenPost }) {
  return (
    <div className="community-list-wrapper">
      <ul className="community-list">
        {posts.map((post) => (
          <li
            key={post.id ?? post.idx ?? Math.random()} // idx 대비 방어
            className="community-row"
            onClick={() => onOpenPost(post.id)}
          >
            <div className="community-thumb">
              {/* ⬇️ CHANGED: 빈 값이면 <img> 렌더링하지 않음 */}
              {post.thumbnail && (
                <img src={post.thumbnail} alt={post.title} />
              )}
            </div>

            <div className="community-row-main">
              <div className="community-row-title-line">
                <span className="community-row-title">{post.title}</span>
                <span className="community-row-count">
                  [{post.commentCount}]
                </span>
              </div>
              <div className="community-row-meta">
                <span className="community-row-writer">
                  {post.writer}
                </span>
                <span className="community-row-dot">·</span>
                <span className="community-row-location">
                  {post.location}
                </span>
              </div>
            </div>

            <div className="community-row-board">{post.board}</div>
            <div className="community-row-time">{post.time}</div>
          </li>
        ))}
      </ul>
    </div>
  );
}

/* ===========================
 *   상세 + 댓글 컴포넌트
 * =========================== */

function CommunityDetail({
  post,
  allPosts,
  comments,
  onBack,
  onAddComment,
  onEdit,
  onDelete,
}) {
  const [commentInput, setCommentInput] = useState("");

  const sameWriterPosts = allPosts.filter(
    (p) => p.writer === post.writer && p.id !== post.id
  );

  const handleSubmit = async (e) => {
    e.preventDefault();
    await onAddComment(post.id, commentInput);
    setCommentInput("");
  };

  return (
    <div className="post-detail-wrapper">
      <button type="button" className="post-detail-back" onClick={onBack}>
        ◀ 목록으로
      </button>

      <div className="post-detail-main">
        <h1 className="post-detail-title">{post.title}</h1>

        {/* 수정/삭제 버튼 */}
        <div style={{ display: "flex", gap: 8, marginBottom: 8 }}>
          <button onClick={onEdit}>수정</button>
          <button onClick={onDelete}>삭제</button>
        </div>

        <div className="post-detail-meta">
          <span className="post-detail-writer">{post.writer}</span>
          <span className="post-detail-dot">·</span>
          <span>{post.board}</span>
          <span className="post-detail-dot">·</span>
          <span>{post.time}</span>
          <span className="post-detail-dot">·</span>
          <span>조회 {post.views}</span>
        </div>

        <div className="post-detail-thumbnail">
          {/* ⬇️ CHANGED: 빈 값이면 렌더링하지 않음 */}
          {post.thumbnail && (
            <img src={post.thumbnail} alt={post.title} />
          )}
        </div>

        {/* ✅ 게시글 본문: body 사용 */}
        <pre className="post-detail-content">{post.body}</pre>
      </div>

      <div className="post-detail-layout-bottom">
        {/* 댓글 */}
        <section className="post-comments">
          <h2 className="post-comments-title">댓글 {comments.length}</h2>

          <ul className="post-comments-list">
            {comments.length === 0 ? (
              <li className="post-comments-empty">첫 댓글을 남겨보세요.</li>
            ) : (
              comments.map((c) => (
                <li key={c.id} className="post-comment-row">
                  <div className="post-comment-header">
                    <span className="post-comment-author">{c.author}</span>
                    <span className="post-comment-dot">·</span>
                    <span className="post-comment-time">
                      {new Date(c.createdAt).toLocaleString()}
                    </span>
                  </div>
                  <div className="post-comment-content">{c.content}</div>
                </li>
              ))
            )}
          </ul>

          <form className="post-comment-form" onSubmit={handleSubmit}>
            <textarea
              className="post-comment-input"
              rows={3}
              placeholder="댓글을 입력하세요."
              value={commentInput}
              onChange={(e) => setCommentInput(e.target.value)}
            />
            <div className="post-comment-actions">
              <button type="submit" className="post-comment-submit">
                등록
              </button>
            </div>
          </form>
        </section>

        {/* 같은 작성자 글 */}
        <aside className="post-writer-more">
          <h3 className="post-writer-more-title">
            {post.writer} 님의 다른 글
          </h3>
          <ul className="post-writer-more-list">
            {sameWriterPosts.length === 0 ? (
              <li className="post-writer-more-empty">
                다른 글이 없습니다.
              </li>
            ) : (
              sameWriterPosts.map((p) => (
                <li key={p.id} className="post-writer-more-row">
                  <span className="post-writer-more-title-text">
                    {p.title}
                  </span>
                  <span className="post-writer-more-count">
                    [{p.commentCount}]
                  </span>
                </li>
              ))
            )}
          </ul>
        </aside>
      </div>
    </div>
  );
}
