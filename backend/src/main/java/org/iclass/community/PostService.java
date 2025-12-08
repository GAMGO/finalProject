package org.iclass.community;

import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;

@Service
public class PostService {

    private final PostRepository repo;

    public PostService(PostRepository repo) {
        this.repo = repo;
    }

    /**
     * 게시글 생성
     */
    public Post create(Post p) {
        return repo.save(p);
    }

    /**
     * 게시글 전체 조회
     */
    public List<Post> all() {
        return repo.findAll();
    }

    /**
     * 게시글 단건 조회
     */
    public Post one(Long idx) {
        return repo.findById(idx)
                .orElseThrow(() -> new IllegalArgumentException("게시글을 찾을 수 없습니다. id=" + idx));
    }

    /**
     * 게시글 수정
     *  - 컨트롤러에서 @PutMapping("/{id}") 에서 호출
     */
    @Transactional
    public Post update(Long id, Post req) {
        Post post = repo.findById(id)
                .orElseThrow(() -> new IllegalArgumentException("게시글을 찾을 수 없습니다. id=" + id));

        // Post 엔티티의 update(...) 시그니처에 맞춰 인자 모두 전달
        post.update(
                req.getTitle(),
                req.getBody(),
                req.getCategory(),
                req.getStoreCategory(),
                req.getLocationText(),
                req.getImageUrl()
        );

        // @Transactional + JPA 변경감지로 DB에 자동 반영
        return post;
    }

    /**
     * 게시글 삭제
     *  - 컨트롤러에서 @DeleteMapping("/{id}") 에서 호출
     */
    public void delete(Long id) {
        if (!repo.existsById(id)) {
            throw new IllegalArgumentException("게시글을 찾을 수 없습니다. id=" + id);
        }
        repo.deleteById(id);
    }
}
