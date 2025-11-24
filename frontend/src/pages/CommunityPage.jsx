// src/pages/CommunityPage.jsx
import React, { useState } from "react";
// ✔ PostFilterBar 지우기
// import PostFilterBar from "../components/community/PostFilterBar";
import PostList from "../components/community/PostList";
import PostEditor from "../components/community/PostEditor";
import "../components/community/Community.css";

const dummyPosts = [
  {
    id: 1,
    type: "제보",
    title: "시청 앞 통닭 트럭 오늘 7시에 온대요",
    content: "어제도 왔는데 줄 미쳤음… 오늘도 온다고 해서 공유합니다.",
    storeCategory: "통닭",
    locationText: "서울 중구 정동길 25 근처",
    createdAt: "5분 전",
    writer: "노점헌터",
    likeCount: 12,
    commentCount: 3,
    imageUrl:
      "https://images.pexels.com/photos/3740193/pexels-photo-3740193.jpeg?auto=compress&cs=tinysrgb&w=800",
  },
  {
    id: 2,
    type: "후기",
    title: "광화문 붕어빵 존맛…",
    content: "팥 듬뿍 + 5개에 3천원. 추울 때 딱이었음.",
    storeCategory: "붕어빵",
    locationText: "광화문역 7번 출구 앞",
    createdAt: "30분 전",
    writer: "붕어덕후",
    likeCount: 5,
    commentCount: 1,
    imageUrl:
      "https://images.pexels.com/photos/4109994/pexels-photo-4109994.jpeg?auto=compress&cs=tinysrgb&w=800",
  },
];

export default function CommunityPage() {
  const [filter, setFilter] = useState("전체");
  const [posts, setPosts] = useState(dummyPosts);
  const [isEditorOpen, setIsEditorOpen] = useState(false);

  const handleFilterChange = (e) => setFilter(e.target.value);
  const handleOpenEditor = () => setIsEditorOpen(true);
  const handleCloseEditor = () => setIsEditorOpen(false);

  const handleCreatePost = (newPost) => {
    const postWithMeta = {
      ...newPost,
      id: Date.now(),
      createdAt: "방금 전",
      likeCount: 0,
      commentCount: 0,
    };
    setPosts((prev) => [postWithMeta, ...prev]);
    setIsEditorOpen(false);
  };

  const filteredPosts =
    filter === "전체" ? posts : posts.filter((p) => p.type === filter);

  return (
    <div className="community-root">
      {/* ✅ 보라색 헤더 제거하고, 콘텐츠 안쪽에 얇은 툴바만 */}
      <div className="community-inner">
        <div className="community-toolbar">
          <h2 className="community-title-plain">나무위키</h2>

          <div className="community-toolbar-right">
            <select
              className="community-category-select"
              value={filter}
              onChange={handleFilterChange}
            >
              <option value="전체">전체</option>
              <option value="제보">제보</option>
              <option value="후기">후기</option>
              <option value="질문">질문</option>
              <option value="잡담">잡담</option>
            </select>

            <button
              type="button"
              className="write-big-btn plain"
              onClick={handleOpenEditor}
            >
              + 글쓰기
            </button>
          </div>
        </div>

        <PostList posts={filteredPosts} />
      </div>

      {isEditorOpen && (
        <PostEditor onClose={handleCloseEditor} onSubmit={handleCreatePost} />
      )}
    </div>
  );
}
