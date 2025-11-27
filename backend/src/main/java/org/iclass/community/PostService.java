package org.iclass.community;

import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class PostService {

    private final PostRepository repo;

    public PostService(PostRepository repo) {
        this.repo = repo;
    }

    public Post create(Post p) {
        return repo.save(p);
    }

    public List<Post> all() {
        return repo.findAll();
    }

    public Post one(Long idx) {
        return repo.findById(idx).orElseThrow();
    }
}
