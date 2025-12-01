package org.iclass.dish;

import jakarta.validation.Valid;
import jakarta.validation.constraints.Positive;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.*;

import java.net.URI;
import java.util.List;

@RestController
@RequestMapping(
        value = "/api/dishes",
        produces = "application/json"
)
@RequiredArgsConstructor
@Validated
public class DishController {

    private final DishService dishService;

    @PostMapping(consumes = "application/json")
    public ResponseEntity<DishResponse> create(@Valid @RequestBody DishRequest request) {
        DishResponse created = dishService.create(request);
        return ResponseEntity
                .created(URI.create("/api/dishes/" + created.getIdx()))
                .body(created);
    }

    @GetMapping
    public ResponseEntity<List<DishResponse>> list() {
        return ResponseEntity.ok(dishService.findAll());
    }

    @GetMapping("/{id}")
    public ResponseEntity<DishResponse> get(@PathVariable("id") @Positive Long idx) {
        return ResponseEntity.ok(dishService.findById(idx));
    }

    @PutMapping(value = "/{id}", consumes = "application/json")
    public ResponseEntity<DishResponse> update(
            @PathVariable("id") @Positive Long idx,
            @Valid @RequestBody DishRequest request
    ) {
        return ResponseEntity.ok(dishService.update(idx, request));
    }

    @DeleteMapping("/{id}")
    public ResponseEntity<Void> delete(@PathVariable("id") @Positive Long idx) {
        dishService.delete(idx);
        return ResponseEntity.noContent().build();
    }
}
