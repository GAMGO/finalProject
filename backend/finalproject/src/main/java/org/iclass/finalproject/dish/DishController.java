package org.iclass.finalproject.dish;

import jakarta.validation.Valid;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.net.URI;
import java.util.List;

@RestController
@RequestMapping("/api/dishes")
public class DishController {

    private final DishService dishService;

    public DishController(DishService dishService) {
        this.dishService = dishService;
    }

    @PostMapping
    public ResponseEntity<DishResponse> create(@Valid @RequestBody DishRequest request) {
        DishResponse created = dishService.create(request);
        return ResponseEntity.created(URI.create("/api/dishes/" + created.getId())).body(created);
    }

    @GetMapping
    public ResponseEntity<List<DishResponse>> list() {
        return ResponseEntity.ok(dishService.findAll());
    }

    @GetMapping("/{id}")
    public ResponseEntity<DishResponse> get(@PathVariable Long id) {
        return ResponseEntity.ok(dishService.findById(id));
    }

    @PutMapping("/{id}")
    public ResponseEntity<DishResponse> update(@PathVariable Long id, @Valid @RequestBody DishRequest request) {
        return ResponseEntity.ok(dishService.update(id, request));
    }

    @DeleteMapping("/{id}")
    public ResponseEntity<Void> delete(@PathVariable Long id) {
        dishService.delete(id);
        return ResponseEntity.noContent().build();
    }
}
