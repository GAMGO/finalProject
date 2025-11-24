import React, { useState } from "react";
import PostFilterBar from "../components/community/PostFilterBar";
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

  const handleChangeFilter = (next) => setFilter(next);
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
      {/* ✅ 위에 보라색 한 줄 헤더 */}
      <div className="community-top">
        <div className="community-top-inner">
          <h2 className="community-title">나무위키</h2>

          <PostFilterBar
            current={filter}
            onChange={handleChangeFilter}
            onClickWrite={handleOpenEditor}
          />
        </div>
      </div>

      {/* 피드 영역 */}
      <div className="community-inner">
        <PostList posts={filteredPosts} />
      </div>

      {isEditorOpen && (
        <PostEditor onClose={handleCloseEditor} onSubmit={handleCreatePost} />
      )}
    </div>
  );
}
