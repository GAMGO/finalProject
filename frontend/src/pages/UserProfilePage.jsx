import React, { useState, useEffect } from "react";
import "./UserProfilePage.css";
import apiClient from "../api/apiClient";

// 생년월일로 나이 계산 (SignupPage랑 같은 컨셉)
const calculateAge = (dobString) => {
  if (!dobString) return null;
  const today = new Date();
  const birthDate = new Date(dobString);
  if (isNaN(birthDate)) return null;

  let age = today.getFullYear() - birthDate.getFullYear();
  const md = today.getMonth() - birthDate.getMonth();
  if (md < 0 || (md === 0 && today.getDate() < birthDate.getDate())) {
    age--;
  }
  return age > 0 ? age : null;
};

export default function UserProfilePage({ onWithdraw }) {
  const [form, setForm] = useState({
    customer_id: "",
    email: "",
    birthDate: "",
    address: "",
  });

  const [emailVerified, setEmailVerified] = useState(false);

  const [isEditing, setIsEditing] = useState(false);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");

  // 비밀번호 변경 섹션
  const [showPasswordPanel, setShowPasswordPanel] = useState(false);
  const [resetCode, setResetCode] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [confirmNewPassword, setConfirmNewPassword] = useState("");
  const [pwMessage, setPwMessage] = useState("");
  const [pwLoading, setPwLoading] = useState(false);

  // -----------------------------
  // 1) 마이페이지 정보 불러오기
  // GET /api/profile/account
  // -----------------------------
  useEffect(() => {
    const fetchAccount = async () => {
      setLoading(true);
      setError("");
      try {
        const res = await apiClient.get("/api/profile/account");
        const data = res.data; // AccountInfoDto

        setForm({
          customer_id: data.id || "",
          email: data.email || "",
          birthDate: (data.birth || "").slice(0, 10), // "YYYY-MM-DD..." -> 앞 10자리만
          address: data.address || "",
        });
        setEmailVerified(Boolean(data.emailVerified));
      } catch (err) {
        console.error("프로필 조회 실패:", err);
        setError(
          err.response?.data?.message ||
            "회원 정보를 불러오는 중 오류가 발생했습니다."
        );
      } finally {
        setLoading(false);
      }
    };

    fetchAccount();
  }, []);

  // -----------------------------
  // 2) 공통 입력 핸들러
  // -----------------------------
  const onChange = (e) => {
    const { name, value } = e.target;
    setForm((p) => ({ ...p, [name]: value }));
  };

  const onClickEdit = () => {
    setIsEditing((v) => !v);
  };

  // -----------------------------
  // 3) 회원정보 저장
  // PUT /api/profile/account
  // -----------------------------
  const onSubmit = async (e) => {
    e.preventDefault();
    if (!isEditing) return;

    setSaving(true);
    setError("");

    const age = calculateAge(form.birthDate);

    const payload = {
      birth: form.birthDate || null,
      age: age,
      address: form.address || null,
    };

    try {
      const res = await apiClient.put("/api/profile/account", payload);
      const data = res.data;

      setForm((prev) => ({
        ...prev,
        birthDate: (data.birth || "").slice(0, 10),
        address: data.address || "",
      }));
      setEmailVerified(Boolean(data.emailVerified));
      setIsEditing(false);
      alert("회원정보가 저장되었습니다.");
    } catch (err) {
      console.error("회원정보 저장 실패:", err);
      setError(
        err.response?.data?.message ||
          "회원정보 저장 중 오류가 발생했습니다."
      );
    } finally {
      setSaving(false);
    }
  };

  // -----------------------------
  // 4) 비밀번호 변경 코드 전송
  // POST /api/profile/account/password/code
  // -----------------------------
  const sendPasswordCode = async () => {
    setPwMessage("");
    setPwLoading(true);
    try {
      await apiClient.post("/api/profile/account/password/code");
      setPwMessage(
        "비밀번호 변경용 인증 코드를 이메일로 전송했습니다. 메일함을 확인하세요."
      );
    } catch (err) {
      console.error("비밀번호 코드 전송 실패:", err);
      setPwMessage(
        err.response?.data?.message ||
          "코드 전송 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요."
      );
    } finally {
      setPwLoading(false);
    }
  };

  // -----------------------------
  // 5) 비밀번호 실제 변경
  // POST /api/profile/account/password/reset
  // -----------------------------
  const submitNewPassword = async (e) => {
    e.preventDefault();
    setPwMessage("");

    if (!resetCode || resetCode.length === 0) {
      setPwMessage("이메일로 받은 인증 코드를 입력해 주세요.");
      return;
    }
    if (!newPassword || newPassword.length < 8) {
      setPwMessage("새 비밀번호는 8자 이상이어야 합니다.");
      return;
    }
    if (newPassword !== confirmNewPassword) {
      setPwMessage("새 비밀번호와 확인이 일치하지 않습니다.");
      return;
    }

    setPwLoading(true);
    try {
      await apiClient.post("/api/profile/account/password/reset", {
        code: resetCode,
        newPassword: newPassword,
      });
      setPwMessage("비밀번호가 변경되었습니다. 다시 로그인해 주세요.");
      setResetCode("");
      setNewPassword("");
      setConfirmNewPassword("");
    } catch (err) {
      console.error("비밀번호 변경 실패:", err);
      setPwMessage(
        err.response?.data?.message ||
          "비밀번호 변경 중 오류가 발생했습니다."
      );
    } finally {
      setPwLoading(false);
    }
  };

  // -----------------------------
  // 6) 렌더링
  // -----------------------------
  if (loading) {
    return (
      <div className="profile-root">
        <div className="profile-inner">
          <div className="profile-card">회원 정보를 불러오는 중...</div>
        </div>
      </div>
    );
  }

  return (
    <div className="profile-root">
      <div className="profile-top">
        <div className="profile-top-inner">
          <h2 className="profile-title">회원정보</h2>
          <button
            className="profile-edit-btn"
            type="button"
            onClick={onClickEdit}
          >
            {isEditing ? "닫기" : "수정하기"}
          </button>
        </div>
      </div>

      <div className="profile-inner">
        {/* 오류 메시지 */}
        {error && <div className="profile-error">{error}</div>}

        {/* 기본 회원 정보 카드 */}
        <form className="profile-card" onSubmit={onSubmit}>
          <div className="profile-row">
            <label>아이디</label>
            <input name="customer_id" value={form.customer_id} disabled />
          </div>

          <div className="profile-row">
            <label>이메일</label>
            <input name="email" value={form.email} disabled />
          </div>

          <div className="profile-row">
            <label>이메일 인증</label>
            <span className="profile-inline-text">
              {emailVerified ? "인증 완료" : "미인증"}
            </span>
          </div>

          <div className="profile-row">
            <label>생년월일</label>
            <input
              type="date"
              name="birthDate"
              value={form.birthDate}
              onChange={onChange}
              disabled={!isEditing}
            />
          </div>
          <div className="profile-row">
            <label>주소</label>
            <input
              name="address"
              value={form.address}
              onChange={onChange}
              disabled={!isEditing}
            />
          </div>

          {isEditing && (
            <div className="profile-actions">
              <button className="profile-save" type="submit" disabled={saving}>
                {saving ? "저장 중..." : "저장"}
              </button>
            </div>
          )}
        </form>

        {/* 비밀번호 변경 카드 */}
        <div className="profile-card profile-password-card">
          <div className="profile-row">
            <label>비밀번호 변경</label>
            <button
              type="button"
              className="profile-edit-btn"
              onClick={() => setShowPasswordPanel((v) => !v)}
            >
              {showPasswordPanel ? "닫기" : "열기"}
            </button>
          </div>

          {showPasswordPanel && (
            <>
              <div className="profile-row">
                <button
                  type="button"
                  className="profile-save"
                  onClick={sendPasswordCode}
                  disabled={pwLoading}
                >
                  {pwLoading ? "코드 전송 중..." : "이메일로 인증 코드 보내기"}
                </button>
              </div>

              <form onSubmit={submitNewPassword}>
                <div className="profile-row">
                  <label>인증 코드</label>
                  <input
                    name="resetCode"
                    value={resetCode}
                    onChange={(e) => setResetCode(e.target.value)}
                  />
                </div>

                <div className="profile-row">
                  <label>새 비밀번호</label>
                  <input
                    type="password"
                    name="newPassword"
                    value={newPassword}
                    onChange={(e) => setNewPassword(e.target.value)}
                  />
                </div>

                <div className="profile-row">
                  <label>새 비밀번호 확인</label>
                  <input
                    type="password"
                    name="confirmNewPassword"
                    value={confirmNewPassword}
                    onChange={(e) => setConfirmNewPassword(e.target.value)}
                  />
                </div>

                {pwMessage && (
                  <div className="profile-pw-message">{pwMessage}</div>
                )}

                <div className="profile-actions">
                  <button
                    className="profile-save"
                    type="submit"
                    disabled={pwLoading}
                  >
                    {pwLoading ? "변경 중..." : "비밀번호 변경"}
                  </button>
                </div>
              </form>
            </>
          )}
        </div>
        {/* 2. 하단에 아주 작게 회원 탈퇴 링크 추가 */}
        <div style={{ marginTop: "30px", textAlign: "right", paddingRight: "20px" }}>
          <span 
            style={{ 
              fontSize: "12px", 
              color: "#999", 
              cursor: "pointer", 
              textDecoration: "underline",
              opacity: 0.7 
            }}
            onClick={onWithdraw} // App.jsx에서 넘겨준 setPage("withdrawal") 실행
          >
            서비스 탈퇴를 원하시나요?
          </span>
        </div>
      </div>
    </div>
  );
}
