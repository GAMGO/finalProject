CREATE DATABASE mydb
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;
USE mydb;

DROP TABLE IF EXISTS chat_message;
DROP TABLE IF EXISTS chat_session;

CREATE TABLE chat_session (
    id           BIGINT       NOT NULL AUTO_INCREMENT,
    user_id      BIGINT       NOT NULL,          -- 지금은 1L 고정으로 쓰는 값
    title        VARCHAR(255) NOT NULL,
    created_at   DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at   DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (id)
) ENGINE=InnoDB
  DEFAULT CHARSET = utf8mb4
  COLLATE = utf8mb4_unicode_ci;
CREATE TABLE chat_message (
    id          BIGINT       NOT NULL AUTO_INCREMENT,
    session_id  BIGINT       NOT NULL,           -- chat_session.id FK
    sender      VARCHAR(20)  NOT NULL,          -- 'USER', 'AI' (SenderType enum)
    content     TEXT         NOT NULL,
    created_at  DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY (id),
    CONSTRAINT fk_chat_message_session
      FOREIGN KEY (session_id)
      REFERENCES chat_session(id)
      ON DELETE CASCADE
) ENGINE=InnoDB
  DEFAULT CHARSET = utf8mb4
  COLLATE = utf8mb4_unicode_ci;
USE mydb;

SELECT * FROM chat_session;
SELECT * FROM chat_message;