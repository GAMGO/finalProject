package org.iclass.finalproject.dish;

import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;

@Service
@Transactional(readOnly = true)
public class DishService {

    private final DishRepository DishRepository;

    public DishService(DishRepository DishRepository) {
        this.DishRepository = DishRepository;
    }

    @Transactional
    public DishResponse create(DishRequest req) {
        if (DishRepository.existsByName(req.getName())) {
            throw new IllegalArgumentException("dish with the same name already exists.");
        }
        dish dish = new dish(req.getName(), req.getDescription(), req.getPrice());
        dish saved = DishRepository.save(dish);
        return DishResponse.from(saved);
    }

    public List<DishResponse> findAll() {
        return DishRepository.findAll().stream().map(DishResponse::from).toList();
    }

    public DishResponse findById(Long id) {
        dish dish = DishRepository.findById(id)
                .orElseThrow(() -> new IllegalArgumentException("dish not found: " + id));
        return DishResponse.from(dish);
    }

    @Transactional
    public DishResponse update(Long id, DishRequest req) {
        dish dish = DishRepository.findById(id)
                .orElseThrow(() -> new IllegalArgumentException("dish not found: " + id));
        dish.setName(req.getName());
        dish.setDescription(req.getDescription());
        dish.setPrice(req.getPrice());
        return DishResponse.from(dish);
    }

    @Transactional
    public void delete(Long id) {
        if (!DishRepository.existsById(id)) {
            throw new IllegalArgumentException("dish not found: " + id);
        }
        DishRepository.deleteById(id);
    }
}
