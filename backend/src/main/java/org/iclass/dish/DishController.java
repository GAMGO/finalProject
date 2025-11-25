package org.iclass.dish;

import jakarta.validation.Valid;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.net.URI;
import java.util.List;

@RestController
@RequestMapping("/api/dishes")
public class DishController {

    private final DishService DishService;

    public DishController(DishService DishService) {
        this.DishService = DishService;
    }

    @PostMapping
    public ResponseEntity<DishResponse> create(@Valid @RequestBody DishRequest request) {
        DishResponse created = DishService.create(request);
        return ResponseEntity.created(URI.create("/api/dishes/" + created.getId())).body(created);
    }

    @GetMapping
    public ResponseEntity<List<DishResponse>> list() {
        return ResponseEntity.ok(DishService.findAll());
    }

    @GetMapping("/{id}")
    public ResponseEntity<DishResponse> get(@PathVariable Long id) {
        return ResponseEntity.ok(DishService.findById(id));
    }

    @PutMapping("/{id}")
    public ResponseEntity<DishResponse> update(@PathVariable Long id, @Valid @RequestBody DishRequest request) {
        return ResponseEntity.ok(DishService.update(id, request));
    }

    @DeleteMapping("/{id}")
    public ResponseEntity<Void> delete(@PathVariable Long id) {
        DishService.delete(id);
        return ResponseEntity.noContent().build();
    }
}
