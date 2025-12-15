import React, { useEffect, useState } from "react";
import apiClient from "../api/apiClient";

export default function AdminStoreChangesPage() {
  const [pending, setPending] = useState([]);
  const [loading, setLoading] = useState(false);
  const [reasonMap, setReasonMap] = useState({});

  const load = async () => {
    setLoading(true);
    try {
      const res = await apiClient.get("/api/admin/store-changes/pending");
      setPending(Array.isArray(res.data) ? res.data : []);
    } catch (e) {
      alert("대기 목록 불러오기 실패 (로그인/권한/토큰 확인)");
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
  }, []);

  const approve = async (id) => {
    if (!window.confirm("승인할까요?")) return;
    try {
      await apiClient.post(`/api/admin/store-changes/${id}/approve`);
      await load();
    } catch (e) {
      alert("승인 실패");
      console.error(e);
    }
  };

  const reject = async (id) => {
    const reason = (reasonMap[id] || "").trim();
    if (!window.confirm("거절할까요?")) return;
    try {
      await apiClient.post(`/api/admin/store-changes/${id}/reject`, { reason });
      await load();
    } catch (e) {
      alert("거절 실패");
      console.error(e);
    }
  };

  const typeLabel = (t) => {
    if (t === "CREATE") return "등록";
    if (t === "UPDATE") return "수정";
    if (t === "DELETE") return "삭제";
    return t;
  };

  return (
    <div style={{ padding: 20 }}>
      <h2 style={{ marginBottom: 12 }}>관리자페이지 - 노점 요청 검수</h2>

      <button onClick={load} disabled={loading} style={{ marginBottom: 12 }}>
        {loading ? "불러오는 중..." : "새로고침"}
      </button>

      {pending.length === 0 ? (
        <div>대기 중인 요청이 없어요.</div>
      ) : (
        <div style={{ display: "grid", gap: 12 }}>
          {pending.map((r) => (
            <div
              key={r.id}
              style={{
                border: "1px solid #e5e7eb",
                borderRadius: 12,
                padding: 12,
                background: "#fff",
              }}
            >
              <div style={{ display: "flex", justifyContent: "space-between", gap: 8 }}>
                <div style={{ fontWeight: 800 }}>
                  요청 #{r.id} / {typeLabel(r.type)} / storeIdx: {r.storeIdx ?? "NEW"}
                </div>
                <div style={{ fontSize: 12, color: "#6b7280" }}>
                  {String(r.requestedAt || "").replace("T", " ").slice(0, 16)}
                </div>
              </div>

              <div style={{ marginTop: 8, fontSize: 13, lineHeight: 1.6 }}>
                <div>이름: {r.newStoreName ?? "-"}</div>
                <div>주소: {r.newStoreAddress ?? "-"}</div>
                <div>
                  좌표: {r.newLat ?? "-"}, {r.newLng ?? "-"}
                </div>
                <div>카테고리ID: {r.newFoodTypeId ?? "-"}</div>
              </div>

              <div style={{ marginTop: 10, display: "flex", gap: 8, alignItems: "center" }}>
                <button onClick={() => approve(r.id)} style={{ padding: "6px 10px" }}>
                  승인
                </button>

                <input
                  placeholder="거절 사유(선택)"
                  value={reasonMap[r.id] || ""}
                  onChange={(e) =>
                    setReasonMap((prev) => ({ ...prev, [r.id]: e.target.value }))
                  }
                  style={{ flex: 1, padding: "6px 10px" }}
                />

                <button onClick={() => reject(r.id)} style={{ padding: "6px 10px" }}>
                  거절
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
