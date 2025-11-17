package com.finalproject.backend.repository;

import com.finalproject.backend.domain.chat.ChatMessage;
import com.finalproject.backend.domain.chat.ChatSession;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;

public interface ChatMessageRepository extends JpaRepository<ChatMessage, Long> {
    List<ChatMessage> findBySessionOrderByCreatedAtAsc(ChatSession session);
}
