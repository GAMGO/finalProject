// src/main/java/com/finalproject/backend/service/ChatService.java
package com.finalproject.backend.service;

import com.finalproject.backend.domain.chat.ChatMessage;
import com.finalproject.backend.domain.chat.ChatSession;
import com.finalproject.backend.domain.chat.SenderType;
import com.finalproject.backend.dto.*;
import com.finalproject.backend.repository.ChatMessageRepository;
import com.finalproject.backend.repository.ChatSessionRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.*;
import java.util.stream.Collectors;

@Service
@RequiredArgsConstructor
@Transactional
public class ChatService {

    private final ChatSessionRepository sessionRepository;
    private final ChatMessageRepository messageRepository;
    private final OpenAiClient openAiClient;

    // 리스트용 시간 포맷 (사이드바에 보일 텍스트)
    private final DateTimeFormatter listFormatter =
            DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm");

    // 로그인 붙기 전이라 userId는 1 고정
    private Long getCurrentUserId() {
        return 1L;
    }

    /* ======================
       세션 만들기 / 목록
       ====================== */

    public ChatSessionDto createSession(String title) {
        ChatSession session = ChatSession.builder()
                .userId(getCurrentUserId())
                .title(title)
                .build();

        ChatSession saved = sessionRepository.save(session);

        return ChatSessionDto.builder()
                .id(saved.getId())
                .title(saved.getTitle())
                // ✅ 사이드바에서 쓸 라벨
                .createdAtLabel(saved.getUpdatedAt().format(listFormatter))
                .build();
    }

    @Transactional(readOnly = true)
    public List<ChatSessionDto> listSessions() {
        Long userId = getCurrentUserId();

        return sessionRepository.findByUserIdOrderByUpdatedAtDesc(userId)
                .stream()
                .map(s -> ChatSessionDto.builder()
                        .id(s.getId())
                        .title(s.getTitle())
                        // ✅ 여기도 동일 포맷으로 내려보내기
                        .createdAtLabel(s.getUpdatedAt().format(listFormatter))
                        .build())
                .collect(Collectors.toList());
    }

    private ChatSession getSessionForCurrentUser(Long sessionId) {
        Long userId = getCurrentUserId();
        ChatSession session = sessionRepository.findById(sessionId)
                .orElseThrow(() -> new RuntimeException("세션을 찾을 수 없습니다."));
        if (!Objects.equals(session.getUserId(), userId)) {
            throw new RuntimeException("권한이 없습니다.");
        }
        return session;
    }

    /* ======================
       세션 삭제
       ====================== */

    public void deleteSession(Long sessionId) {
        ChatSession session = getSessionForCurrentUser(sessionId);

        // 1) 먼저 해당 세션의 메시지들 삭제
        messageRepository.deleteBySession(session);

        // 2) 세션 삭제
        sessionRepository.delete(session);
    }

    /* ======================
       메시지 관련
       ====================== */

    @Transactional(readOnly = true)
    public List<ChatMessageDto> getMessages(Long sessionId) {
        ChatSession session = getSessionForCurrentUser(sessionId);
        return messageRepository.findBySessionOrderByCreatedAtAsc(session)
                .stream()
                .map(m -> ChatMessageDto.builder()
                        .id(m.getId())
                        .sender(m.getSender().name())
                        .content(m.getContent())
                        .build())
                .collect(Collectors.toList());
    }

    public ChatReplyDto processMessage(Long sessionId, String content) {
        ChatSession session = getSessionForCurrentUser(sessionId);

        // 1) 유저 메시지 저장
        ChatMessage userMsg = ChatMessage.builder()
                .session(session)
                .sender(SenderType.USER)
                .content(content)
                .build();
        userMsg = messageRepository.save(userMsg);

        // 2) GPT에 보낼 전체 대화 구성
        List<ChatMessage> history =
                messageRepository.findBySessionOrderByCreatedAtAsc(session);

        List<Map<String, String>> messages = new ArrayList<>();

        String systemPrompt = """
                너는 엑셀 양식 설계를 도와주는 한국어 AI 비서야.
                - 사용자의 업무 설명을 듣고, 먼저 어떤 열(컬럼)과 형식으로 만들지 제안해.
                - 항상 2단계로 진행해:

                1단계: 형식 제안
                 - 사용자의 요구를 한 줄로 요약하고
                 - 추천 컬럼 목록을 bullet 형식으로 제안하고
                 - 마지막에는 반드시 "이 형식으로 만들어드릴까요?"라고 물어봐.

                2단계: 사용자가 "네", "이대로", "좋아요"처럼 동의하면
                 - "그럼 이 형식으로 엑셀을 생성하겠습니다."라고 대답해.
                 - 그리고 어떤 엑셀이 만들어질지 간단히 다시 정리해 줘.

                쓸데없는 잡담은 하지 말고, 업무에만 집중해.
                """;

        messages.add(Map.of("role", "system", "content", systemPrompt));

        for (ChatMessage m : history) {
            String role = (m.getSender() == SenderType.USER) ? "user" : "assistant";
            messages.add(Map.of("role", role, "content", m.getContent()));
        }

        // 이번에 새로 들어온 유저 메시지도 마지막에 추가
        messages.add(Map.of("role", "user", "content", content));

        // 3) GPT 호출
        String assistantText = openAiClient.chat(messages);

        // 4) AI 메시지 저장
        ChatMessage aiMsg = ChatMessage.builder()
                .session(session)
                .sender(SenderType.AI)
                .content(assistantText)
                .build();
        aiMsg = messageRepository.save(aiMsg);

        // 세션 업데이트 시간 갱신
        session.setUpdatedAt(LocalDateTime.now());

        return ChatReplyDto.builder()
                .userMessage(ChatMessageDto.builder()
                        .id(userMsg.getId())
                        .sender("USER")
                        .content(userMsg.getContent())
                        .build())
                .assistantMessage(ChatMessageDto.builder()
                        .id(aiMsg.getId())
                        .sender("AI")
                        .content(aiMsg.getContent())
                        .build())
                .build();
    }
}
