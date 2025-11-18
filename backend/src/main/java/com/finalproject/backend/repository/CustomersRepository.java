package com.finalproject.backend.repository;

import org.springframework.data.jpa.repository.JpaRepository;

import com.finalproject.backend.entity.Customers;

public interface CustomersRepository extends JpaRepository<Customers, Long> {
  boolean existsByCustomersId(String customersId);
}
