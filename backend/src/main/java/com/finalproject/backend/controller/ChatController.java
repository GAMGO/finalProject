package com.finalproject.backend.controller;

import com.finalproject.backend.dto.*;
import com.finalproject.backend.service.ChatService;
import lombok.RequiredArgsConstructor;
import org.springframework.core.io.Resource;
import org.springframework.core.io.UrlResource;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.net.URLEncoder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;

@RestController
@RequestMapping("/api/chat")
@RequiredArgsConstructor
@CrossOrigin(origins = "http://localhost:5173")
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

    @DeleteMapping("/sessions/{id}")
    public void deleteSession(@PathVariable Long id) {
        chatService.deleteSession(id);
    }

    @PostMapping("/sessions/{id}/upload")
    public ChatMessageDto uploadFile(@PathVariable Long id,
                                     @RequestParam("file") MultipartFile file,
                                     @RequestParam(value = "type",
                                                   required = false,
                                                   defaultValue = "FILE")
                                     String type) {
        return chatService.uploadFile(id, file, type);
    }


@GetMapping("/files/{storedName}")
public ResponseEntity<Resource> downloadFile(@PathVariable String storedName) throws IOException {
    String rootDir = System.getProperty("user.dir");
    Path uploadPath = Paths.get(rootDir, "uploads").normalize();
    Path filePath = uploadPath.resolve(storedName).normalize();

    if (!Files.exists(filePath)) {
        return ResponseEntity.notFound().build();
    }

    Resource resource = new UrlResource(filePath.toUri());

    String originalName = storedName;
    int idx = storedName.indexOf('_');
    if (idx != -1 && idx < storedName.length() - 1) {
        originalName = storedName.substring(idx + 1);
    }

    String encodedName = URLEncoder.encode(originalName, StandardCharsets.UTF_8)
                                   .replace("+", "%20");

    return ResponseEntity.ok()
            .header(HttpHeaders.CONTENT_DISPOSITION,
                    "attachment; filename=\"" + encodedName + "\"")
            .contentType(MediaType.APPLICATION_OCTET_STREAM)
            .body(resource);
}

}
