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
import org.springframework.util.StringUtils;
import org.springframework.web.multipart.MultipartFile;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
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

        // 리스트용 시간 포맷
        private final DateTimeFormatter listFormatter = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm");

        private Long getCurrentUserId() {
                return 1L;
        }

        /*
         * ======================
         * 세션 만들기 / 목록
         * ======================
         */

        public ChatSessionDto createSession(String title) {
                ChatSession session = ChatSession.builder()
                                .userId(getCurrentUserId())
                                .title(title)
                                .build();

                ChatSession saved = sessionRepository.save(session);

                return ChatSessionDto.builder()
                                .id(saved.getId())
                                .title(saved.getTitle())
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

        /*
         * ======================
         * 세션 삭제
         * ======================
         */

        public void deleteSession(Long sessionId) {
                ChatSession session = getSessionForCurrentUser(sessionId);
                messageRepository.deleteBySession(session);
                sessionRepository.delete(session);
        }

        /*
         * ======================
         * 메시지 조회
         * ======================
         */

        @Transactional(readOnly = true)
        public List<ChatMessageDto> getMessages(Long sessionId) {
                ChatSession session = getSessionForCurrentUser(sessionId);
                return messageRepository.findBySessionOrderByCreatedAtAsc(session)
                                .stream()
                                .map(m -> ChatMessageDto.builder()
                                                .id(m.getId())
                                                .sender(m.getSender().name())
                                                .content(m.getContent())
                                                .fileName(m.getFileName())
                                                .fileUrl(m.getFileUrl())
                                                .fileType(m.getFileType())
                                                .build())
                                .collect(Collectors.toList());
        }

        /*
         * ======================
         * 텍스트 메시지
         * ======================
         */

        public ChatReplyDto processMessage(Long sessionId, String content) {
                ChatSession session = getSessionForCurrentUser(sessionId);

                // 1) 유저 메시지 저장
                ChatMessage userMsg = ChatMessage.builder()
                                .session(session)
                                .sender(SenderType.USER)
                                .content(content)
                                .build();
                userMsg = messageRepository.save(userMsg);

                // 2) GPT 대화 구성
                List<ChatMessage> history = messageRepository.findBySessionOrderByCreatedAtAsc(session);

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

      // src/main/java/com/finalproject/backend/service/ChatService.java

public ChatMessageDto uploadFile(Long sessionId, MultipartFile file, String type) {
    ChatSession session = getSessionForCurrentUser(sessionId);

    if (file == null || file.isEmpty()) {
        throw new RuntimeException("파일이 없습니다.");
    }

    try {
        String rootDir = System.getProperty("user.dir");
        Path uploadPath = Paths.get(rootDir, "uploads");

        if (!Files.exists(uploadPath)) {
            Files.createDirectories(uploadPath);
        }

        String originalName = StringUtils.cleanPath(
                Objects.requireNonNull(file.getOriginalFilename()));
        String storedName = UUID.randomUUID() + "_" + originalName;

        Path target = uploadPath.resolve(storedName);
        file.transferTo(target.toFile());

        // ✅ 브라우저에서 호출할 다운로드 API 경로
        String fileUrl = "/api/chat/files/" + storedName;

        ChatMessage msg = ChatMessage.builder()
                .session(session)
                .sender(SenderType.USER)
                .content("")
                .fileName(originalName)
                .fileUrl(fileUrl)
                .fileType(type)
                .build();

        msg = messageRepository.save(msg);
        session.setUpdatedAt(LocalDateTime.now());

        return ChatMessageDto.builder()
                .id(msg.getId())
                .sender("USER")
                .content("")
                .fileName(msg.getFileName())
                .fileUrl(msg.getFileUrl())
                .fileType(msg.getFileType())
                .build();

    } catch (Exception e) {
        throw new RuntimeException("파일 업로드 실패", e);
    }
}

}
