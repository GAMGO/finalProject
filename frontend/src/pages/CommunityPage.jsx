import React, { useState } from "react";
import PostList from "../components/community/PostList";
import PostEditor from "../components/community/PostEditor";
import "../components/community/Community.css";

// ✅ 목데이터 (적당히 많이 넣어둠)
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
      "https://images.pexels.com/photos/291528/pexels-photo-291528.jpeg?auto=compress&cs=tinysrgb&w=1200",
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
      "https://images.pexels.com/photos/460537/pexels-photo-460537.jpeg?auto=compress&cs=tinysrgb&w=1200",
  },
  {
    id: 3,
    type: "후기",
    title: "홍대 버블티 트럭, 토핑 미쳤음",
    content: "펄 + 치즈폼 조합 ㄹㅇ 인생 버블티.",
    storeCategory: "디저트",
    locationText: "홍대입구역 9번 출구 앞",
    createdAt: "1시간 전",
    writer: "버블덕후",
    likeCount: 18,
    commentCount: 4,
    imageUrl:
      "https://images.pexels.com/photos/3731474/pexels-photo-3731474.jpeg?auto=compress&cs=tinysrgb&w=1200",
  },
  {
    id: 4,
    type: "질문",
    title: "신촌 쪽 닭강정 트럭 어디 갔나요?",
    content: "작년 겨울마다 있던 주황색 트럭 못 보겠음… 혹시 아는 사람?",
    storeCategory: "닭강정",
    locationText: "신촌역 근처",
    createdAt: "2시간 전",
    writer: "찾는중",
    likeCount: 3,
    commentCount: 6,
    imageUrl:
      "https://images.pexels.com/photos/4109136/pexels-photo-4109136.jpeg?auto=compress&cs=tinysrgb&w=1200",
  },
  {
    id: 5,
    type: "제보",
    title: "건대입구 김밥 튀김 노점 떴어요",
    content: "오늘만 한다고 해서 공유. 가격도 싸고 양 많음.",
    storeCategory: "분식",
    locationText: "건대입구역 5번 출구 앞",
    createdAt: "3시간 전",
    writer: "건대러",
    likeCount: 9,
    commentCount: 2,
    imageUrl:
      "https://images.pexels.com/photos/884600/pexels-photo-884600.jpeg?auto=compress&cs=tinysrgb&w=1200",
  },
  {
    id: 6,
    type: "후기",
    title: "잠실 야구장 앞 핫도그 트럭 인정",
    content: "경기 보기 전에 한 입 딱 먹어줘야 함.",
    storeCategory: "핫도그",
    locationText: "잠실 야구장 3루 쪽",
    createdAt: "어제",
    writer: "야구봐",
    likeCount: 21,
    commentCount: 5,
    imageUrl:
      "https://images.pexels.com/photos/461198/pexels-photo-461198.jpeg?auto=compress&cs=tinysrgb&w=1200",
  },
  {
    id: 7,
    type: "후기",
    title: "부산 서면 야시장 닭꼬치 줄 너무 길다",
    content: "맛은 있는데 최소 20분은 서야 함 ㅠ",
    storeCategory: "닭꼬치",
    locationText: "부산 서면 야시장",
    createdAt: "어제",
    writer: "부산인",
    likeCount: 7,
    commentCount: 1,
    imageUrl:
      "https://images.pexels.com/photos/106343/pexels-photo-106343.jpeg?auto=compress&cs=tinysrgb&w=1200",
  },
  {
    id: 8,
    type: "제보",
    title: "연남동 타코 트럭 오늘 신메뉴 나왔다",
    content: "새우 타코 나왔다길래 일단 올려봄.",
    storeCategory: "타코",
    locationText: "연남동 동진시장 근처",
    createdAt: "2일 전",
    writer: "타코타코",
    likeCount: 11,
    commentCount: 3,
    imageUrl:
      "https://images.pexels.com/photos/461198/pexels-photo-461198.jpeg?auto=compress&cs=tinysrgb&w=1200",
  },
  {
    id: 9,
    type: "질문",
    title: "을지로 직화구이 노점 아직 있나요?",
    content: "퇴근하고 가고 싶은데 요즘도 나오시는지 궁금함.",
    storeCategory: "구이",
    locationText: "을지로3가역 근처",
    createdAt: "3일 전",
    writer: "직장인",
    likeCount: 2,
    commentCount: 2,
    imageUrl:
      "https://images.pexels.com/photos/4109133/pexels-photo-4109133.jpeg?auto=compress&cs=tinysrgb&w=1200",
  },
  {
    id: 10,
    type: "후기",
    title: "대구 서문시장 야식 코스 추천",
    content: "어묵 → 납작만두 → 닭강정 루트 추천함.",
    storeCategory: "야시장",
    locationText: "대구 서문시장",
    createdAt: "4일 전",
    writer: "대구야식러",
    likeCount: 17,
    commentCount: 7,
    imageUrl:
      "https://images.pexels.com/photos/1059943/pexels-photo-1059943.jpeg?auto=compress&cs=tinysrgb&w=1200",
  },
  {
    id: 11,
    type: "잡담",
    title: "오늘 날씨에 붕어빵이냐 호떡이냐",
    content: "다들 뭐 드실 건가요 ㅋㅋ",
    storeCategory: "기타",
    locationText: "서울 어딘가",
    createdAt: "4일 전",
    writer: "고민중",
    likeCount: 4,
    commentCount: 9,
    imageUrl:
      "https://images.pexels.com/photos/4109994/pexels-photo-4109994.jpeg?auto=compress&cs=tinysrgb&w=1200",
  },
  {
    id: 12,
    type: "제보",
    title: "신사 가로수길 커피 트럭 떴어요",
    content: "라떼 맛있고 머그컵에 주셔서 예쁨.",
    storeCategory: "커피",
    locationText: "신사 가로수길 정문 앞",
    createdAt: "5일 전",
    writer: "카페인중독",
    likeCount: 6,
    commentCount: 1,
    imageUrl:
      "https://images.pexels.com/photos/373888/pexels-photo-373888.jpeg?auto=compress&cs=tinysrgb&w=1200",
  },
];

const POSTS_PER_PAGE = 6;

export default function CommunityPage() {
  const [filter, setFilter] = useState("전체");
  const [posts, setPosts] = useState(dummyPosts);
  const [isEditorOpen, setIsEditorOpen] = useState(false);
  const [currentPage, setCurrentPage] = useState(1);

  const handleFilterChange = (e) => {
    setFilter(e.target.value);
    setCurrentPage(1); // 필터 바꾸면 1페이지로
  };

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
    setCurrentPage(1);
  };

  // 필터 적용
  const filteredPosts =
    filter === "전체" ? posts : posts.filter((p) => p.type === filter);

  // 페이지네이션 계산
  const totalPages = Math.max(
    1,
    Math.ceil(filteredPosts.length / POSTS_PER_PAGE)
  );
  const startIdx = (currentPage - 1) * POSTS_PER_PAGE;
  const pagePosts = filteredPosts.slice(startIdx, startIdx + POSTS_PER_PAGE);

  const handlePageChange = (page) => {
    if (page < 1 || page > totalPages) return;
    setCurrentPage(page);
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  return (
    <div className="community-root">
      <div className="community-inner">
        {/* 상단 필터/버튼 */}
        <div className="community-toolbar">
          <h2 className="community-title-plain">커뮤니티</h2>

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

        {/* 카드 6개씩 */}
        <PostList posts={pagePosts} />

        {/* 페이지네이션 */}
        <div className="pagination">
          <button
            className="page-arrow"
            onClick={() => handlePageChange(currentPage - 1)}
            disabled={currentPage === 1}
          >
            ‹
          </button>

          {Array.from({ length: totalPages }, (_, i) => {
            const page = i + 1;
            return (
              <button
                key={page}
                className={
                  "page-btn" +
                  (page === currentPage ? " page-btn-active" : "")
                }
                onClick={() => handlePageChange(page)}
              >
                {page}
              </button>
            );
          })}

          <button
            className="page-arrow"
            onClick={() => handlePageChange(currentPage + 1)}
            disabled={currentPage === totalPages}
          >
            ›
          </button>
        </div>
      </div>

      {isEditorOpen && (
        <PostEditor onClose={handleCloseEditor} onSubmit={handleCreatePost} />
      )}
    </div>
  );
}
