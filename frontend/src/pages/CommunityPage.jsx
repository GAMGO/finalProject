// src/pages/CommunityPage.jsx
import React, { useState } from "react";
import "../components/community/Community.css";

// ✅ 노점 목데이터 (DCinside 스타일)
const stallPosts = [
  {
    id: 1,
    title: "시청 앞 통닭 트럭 오늘 7시에 온대요",
    writer: "노점헌터",
    board: "서울 노점 제보",
    time: "13:50",
    commentCount: 88,
    views: 1240,
    category: "제보",
    location: "서울 중구 시청역 2번 출구 앞",
    thumbnail:
      "https://images.pexels.com/photos/461198/pexels-photo-461198.jpeg?auto=compress&cs=tinysrgb&w=400",
    content:
      "어제도 왔던 통닭 트럭인데 오늘도 7시쯤 온다고 해서 제보합니다.\n\n가격은 한 마리 15,000원, 감자튀김 추가 3,000원이고, 줄 금방 길어지니 미리 가는 거 추천!",
  },
  {
    id: 2,
    title: "광화문 붕어빵 5개 3천원 파는 곳 후기",
    writer: "붕어덕후",
    board: "간식·디저트",
    time: "13:35",
    commentCount: 32,
    views: 830,
    category: "후기",
    location: "광화문역 7번 출구 앞",
    thumbnail:
      "https://images.pexels.com/photos/4109994/pexels-photo-4109994.jpeg?auto=compress&cs=tinysrgb&w=400",
    content:
      "팥 듬뿍 + 크기도 커서 가성비 상당합니다.\n\n다만 줄이 조금 길고, 카드 결제는 안 됩니다.",
  },
  {
    id: 3,
    title: "연남동 타코 트럭 신메뉴 나왔대요",
    writer: "타코타코",
    board: "연남·홍대",
    time: "13:20",
    commentCount: 10,
    views: 542,
    category: "제보",
    location: "연남동 동진시장 근처",
    thumbnail:
      "https://images.pexels.com/photos/461198/pexels-photo-461198.jpeg?auto=compress&cs=tinysrgb&w=400",
    content:
      "새우 타코 나왔다고 해서 점심에 먹어봤는데 소스가 살짝 매콤달콤.\n\n오늘 내일만 한정 메뉴라길래 공유합니다.",
  },
  {
    id: 4,
    title: "신촌 닭강정 트럭 요즘도 나오나요?",
    writer: "찾는중",
    board: "질문·찾기",
    time: "13:05",
    commentCount: 49,
    views: 910,
    category: "질문",
    location: "신촌역 현대백화점 쪽",
    thumbnail:
      "https://images.pexels.com/photos/4109136/pexels-photo-4109136.jpeg?auto=compress&cs=tinysrgb&w=400",
    content:
      "작년 겨울마다 보이던 주황색 트럭이 안 보이네요.\n\n혹시 요즘도 나오는지, 요일이 바뀐 건지 아시는 분 계신가요?",
  },
  {
    id: 5,
    title: "서면 야시장 닭꼬치 줄 미쳤네요ㅋㅋ",
    writer: "부산인",
    board: "부산 노점",
    time: "12:45",
    commentCount: 16,
    views: 678,
    category: "후기",
    location: "부산 서면 야시장",
    thumbnail:
      "https://images.pexels.com/photos/106343/pexels-photo-106343.jpeg?auto=compress&cs=tinysrgb&w=400",
    content:
      "맛은 있는데 최소 20분은 서야 먹을 수 있습니다…\n\n줄 길이 감안하고 가세요 ㅠ",
  },
];

export default function CommunityPage() {
  // 리스트 / 상세 보기 상태
  const [selectedPostId, setSelectedPostId] = useState(null);

  // 댓글 상태: { [postId]: [{ id, author, text, createdAt }] }
  const [commentsByPost, setCommentsByPost] = useState({});

  const selectedPost = stallPosts.find((p) => p.id === selectedPostId) || null;

  const handleOpenPost = (postId) => {
    setSelectedPostId(postId);
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  const handleBackToList = () => {
    setSelectedPostId(null);
  };

  const handleAddComment = (postId, text) => {
    if (!text.trim()) return;
    setCommentsByPost((prev) => {
      const list = prev[postId] || [];
      const newComment = {
        id: Date.now(),
        author: "익명",
        text: text.trim(),
        createdAt: "방금 전",
      };
      return {
        ...prev,
        [postId]: [...list, newComment],
      };
    });
  };

  return (
    <div className="community-root">
      {/* 상단 중앙 보라색 입력창 (검색/글쓰기 느낌) */}
      <div className="community-input-wrapper">
        <input
          className="community-main-input"
          type="text"
          placeholder="노점 제보, 후기, 장소를 입력해 보세요"
        />
      </div>

      {/* 리스트 / 상세 모드 분기 */}
      {!selectedPost ? (
        <CommunityList posts={stallPosts} onOpenPost={handleOpenPost} />
      ) : (
        <CommunityDetail
          post={selectedPost}
          allPosts={stallPosts}
          comments={commentsByPost[selectedPost.id] || []}
          onBack={handleBackToList}
          onAddComment={handleAddComment}
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
            key={post.id}
            className="community-row"
            onClick={() => onOpenPost(post.id)}
          >
            <div className="community-thumb">
              <img src={post.thumbnail} alt={post.title} />
            </div>

            <div className="community-row-main">
              <div className="community-row-title-line">
                <span className="community-row-title">{post.title}</span>
                <span className="community-row-count">
                  [{post.commentCount}]
                </span>
              </div>
              <div className="community-row-meta">
                <span className="community-row-writer">{post.writer}</span>
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
}) {
  const [commentInput, setCommentInput] = useState("");

  const sameWriterPosts = allPosts.filter(
    (p) => p.writer === post.writer && p.id !== post.id
  );

  const handleSubmit = (e) => {
    e.preventDefault();
    onAddComment(post.id, commentInput);
    setCommentInput("");
  };

  return (
    <div className="post-detail-wrapper">
      <button type="button" className="post-detail-back" onClick={onBack}>
        ◀ 목록으로
      </button>

      <div className="post-detail-main">
        <h1 className="post-detail-title">{post.title}</h1>

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
          <img src={post.thumbnail} alt={post.title} />
        </div>

        <pre className="post-detail-content">{post.content}</pre>
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
                    <span className="post-comment-time">{c.createdAt}</span>
                  </div>
                  <div className="post-comment-text">{c.text}</div>
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
              <li className="post-writer-more-empty">다른 글이 없습니다.</li>
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
