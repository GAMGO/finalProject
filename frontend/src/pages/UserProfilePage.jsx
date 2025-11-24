import React, { useState } from "react";
import "./UserProfilePage.css";

export default function UserProfilePage() {
  const [form, setForm] = useState({
    customerId: "testuser",
    email: "test@example.com",
    name: "익명",
    birthDate: "1999-01-01",
    gender: "남",
    address: "서울 어딘가",
  });

  const [isEditing, setIsEditing] = useState(false);

  const onChange = (e) => {
    const { name, value } = e.target;
    setForm((p) => ({ ...p, [name]: value }));
  };

  const onClickEdit = () => setIsEditing((v) => !v);

  const onSubmit = (e) => {
    e.preventDefault();
    setIsEditing(false);
    // TODO: API 붙일 자리
    console.log("save profile", form);
  };

  return (
    <div className="profile-root">
      <div className="profile-top">
        <div className="profile-top-inner">
          <h2 className="profile-title">회원정보</h2>
          <button className="profile-edit-btn" type="button" onClick={onClickEdit}>
            {isEditing ? "닫기" : "수정하기"}
          </button>
        </div>
      </div>

      <div className="profile-inner">
        <form className="profile-card" onSubmit={onSubmit}>
          <div className="profile-row">
            <label>아이디</label>
            <input name="customerId" value={form.customerId} disabled />
          </div>

          <div className="profile-row">
            <label>이메일</label>
            <input
              name="email"
              value={form.email}
              onChange={onChange}
              disabled={!isEditing}
            />
          </div>

          <div className="profile-row">
            <label>이름</label>
            <input
              name="name"
              value={form.name}
              onChange={onChange}
              disabled={!isEditing}
            />
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
            <label>성별</label>
            <select
              name="gender"
              value={form.gender}
              onChange={onChange}
              disabled={!isEditing}
            >
              <option value="남">남</option>
              <option value="여">여</option>
              <option value="기타">기타</option>
            </select>
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
              <button className="profile-save" type="submit">
                저장
              </button>
            </div>
          )}
        </form>
      </div>
    </div>
  );
}
