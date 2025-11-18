package com.finalproject.backend.entity;

import java.time.LocalDateTime;

import jakarta.persistence.CascadeType;
import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.OneToOne;
import jakarta.persistence.Table;
import lombok.Data;

// Customers.java
@Data
@Entity
@Table(name = "CUSTOMERS")
public class Customers {
  @Id
  @GeneratedValue(strategy = GenerationType.IDENTITY)
  private Long idx;

  @Column(nullable = false, unique = true, length = 100)
  private String customersId;

  @Column(nullable = true, unique = true, length = 255)
  private String email; // 🔥 반드시 추가해야 함!!

  @Column(nullable = false, length = 255)
  private String password;

  private Integer emailVerified;

  private String emailVerificationToken;

  private LocalDateTime emailVerificationExpires;

  @OneToOne(mappedBy = "customer", cascade = CascadeType.ALL)
  private UserProfile userProfile;
}