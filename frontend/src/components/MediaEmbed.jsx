// src/components/MediaEmbed.jsx
import React from "react";

/** 유튜브 watch / shorts / youtu.be 링크 → embed URL */
const getYoutubeEmbedUrl = (url) => {
  if (!url) return null;
  try {
    const u = new URL(url);
    let id = "";

    if (u.hostname.includes("youtu.be")) {
      // https://youtu.be/VIDEO_ID
      id = u.pathname.slice(1);
    } else if (u.hostname.includes("youtube.com")) {
      if (u.searchParams.get("v")) {
        // https://www.youtube.com/watch?v=VIDEO_ID
        id = u.searchParams.get("v");
      } else if (u.pathname.startsWith("/embed/")) {
        // https://www.youtube.com/embed/VIDEO_ID
        id = u.pathname.split("/embed/")[1];
      } else if (u.pathname.startsWith("/shorts/")) {
        // https://www.youtube.com/shorts/VIDEO_ID
        id = u.pathname.split("/shorts/")[1];
      }
    }

    id = id.split(/[?&]/)[0];
    if (!id) return null;
    return `https://www.youtube.com/embed/${id}`;
  } catch {
    return null;
  }
};

const isDirectVideoFile = (url) => {
  if (!url) return false;
  const base = url.split("?")[0].toLowerCase();
  return /\.(mp4|webm|ogg)$/i.test(base);
};

const isBlobUrl = (url) => typeof url === "string" && url.startsWith("blob:");

export default function MediaEmbed({ url, poster, className = "" }) {
  if (!url) return null;

  const youtube = getYoutubeEmbedUrl(url);
  const isFile = isBlobUrl(url) || isDirectVideoFile(url);

  let content = null;

  if (youtube) {
    // 유튜브
    content = (
      <iframe
        src={youtube}
        title="youtube-video"
        className={`media-embed-iframe ${className}`}
        frameBorder="0"
        allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
        allowFullScreen
      />
    );
  } else if (isFile) {
    // blob / 직접 영상 파일
    content = (
      <video
        src={url}
        className={`media-embed-video ${className}`}
        controls
        poster={poster}
      />
    );
  } else {
    // 그 외 (네이버, 카카오 등) → iframe 시도 + 새창 링크 제공
    content = (
      <iframe
        src={url}
        title="external-video"
        className={`media-embed-iframe ${className}`}
        frameBorder="0"
        allow="autoplay; fullscreen"
        allowFullScreen
        // 너무 막힌 사이트 대비해서 sandbox 완화
        sandbox="allow-same-origin allow-scripts allow-popups allow-forms allow-presentation"
      />
    );
  }

  return (
    <div className="media-embed-wrapper">
      {content}
      <a
        href={url}
        target="_blank"
        rel="noreferrer"
        className="media-embed-open-link"
      >
        새 창에서 영상 보기 ↗
      </a>
    </div>
  );
}
