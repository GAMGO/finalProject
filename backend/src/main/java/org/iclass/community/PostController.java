// src/main/java/org/iclass/community/PostController.java
package org.iclass.community;

import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.http.HttpStatus;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/posts")
@RequiredArgsConstructor
public class PostController {

    private final PostService service;

    /**
     * 게시글 생성
     */
    @PostMapping
    @ResponseStatus(HttpStatus.CREATED)
    public Post create(@Valid @RequestBody Post req) {
        return service.create(req);
    }

    /**
     * 게시글 전체 목록
     */
    @GetMapping
    public List<Post> all() {
        return service.all();
    }

    /**
     * 게시글 단건 조회
     */
    @GetMapping("/{id}")
    public Post one(@PathVariable("id") Long idx) {
        return service.one(idx);
    }

    /**
     * 게시글 수정
     */
    @PutMapping("/{id}")
    public Post update(
            @PathVariable("id") Long idx,
            @Valid @RequestBody Post req
    ) {
        return service.update(idx, req);
    }

    /**
     * 게시글 삭제
     */
    @DeleteMapping("/{id}")
    @ResponseStatus(HttpStatus.NO_CONTENT)
    public void delete(@PathVariable("id") Long idx) {
        service.delete(idx);
    }
}
