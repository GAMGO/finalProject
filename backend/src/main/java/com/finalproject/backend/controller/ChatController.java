package com.finalproject.backend.controller;

import com.finalproject.backend.dto.*;
import com.finalproject.backend.service.ChatService;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/chat")
@RequiredArgsConstructor
@CrossOrigin(origins = "http://localhost:5173") // 리액트 dev 서버
public class ChatController {

    private final ChatService chatService;

    @PostMapping("/sessions")
    public ChatSessionDto createSession(@RequestBody CreateSessionReq req) {
        return chatService.createSession(req.getTitle());
    }

    @GetMapping("/sessions")
    public List<ChatSessionDto> listSessions() {
        return chatService.listSessions();
    }

    @GetMapping("/sessions/{id}/messages")
    public List<ChatMessageDto> getMessages(@PathVariable Long id) {
        return chatService.getMessages(id);
    }

    @PostMapping("/sessions/{id}/messages")
    public ChatReplyDto sendMessage(@PathVariable Long id,
                                    @RequestBody SendMessageReq req) {
        return chatService.processMessage(id, req.getContent());
    }
}
