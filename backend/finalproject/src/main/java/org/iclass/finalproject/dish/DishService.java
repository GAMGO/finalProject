package org.iclass.finalproject.dish;

import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;

@Service
@Transactional(readOnly = true)
public class DishService {

    private final DishRepository dishRepository;

    public DishService(DishRepository dishRepository) {
        this.dishRepository = dishRepository;
    }

    @Transactional
    public DishResponse create(DishRequest req) {
        if (dishRepository.existsByName(req.getName())) {
            throw new IllegalArgumentException("Dish with the same name already exists.");
        }
        Dish dish = new Dish(req.getName(), req.getDescription(), req.getPrice());
        Dish saved = dishRepository.save(dish);
        return DishResponse.from(saved);
    }

    public List<DishResponse> findAll() {
        return dishRepository.findAll().stream().map(DishResponse::from).toList();
    }

    public DishResponse findById(Long id) {
        Dish dish = dishRepository.findById(id)
                .orElseThrow(() -> new IllegalArgumentException("Dish not found: " + id));
        return DishResponse.from(dish);
    }

    @Transactional
    public DishResponse update(Long id, DishRequest req) {
        Dish dish = dishRepository.findById(id)
                .orElseThrow(() -> new IllegalArgumentException("Dish not found: " + id));
        dish.setName(req.getName());
        dish.setDescription(req.getDescription());
        dish.setPrice(req.getPrice());
        return DishResponse.from(dish);
    }

    @Transactional
    public void delete(Long id) {
        if (!dishRepository.existsById(id)) {
            throw new IllegalArgumentException("Dish not found: " + id);
        }
        dishRepository.deleteById(id);
    }
}
