package com.finalproject.backend.entity;

import jakarta.persistence.Entity;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.JoinColumn;
import jakarta.persistence.OneToOne;
import jakarta.persistence.Table;
import lombok.Data;

@Data
@Entity
@Table(name = "USER_PROFILES")
public class UserProfile {
  @Id
  @GeneratedValue(strategy = GenerationType.IDENTITY)
  private Long idx;

  private String name;
  private String birth;
  private Integer gender;

  @OneToOne
  @JoinColumn(name = "CUSTOMER_ID", referencedColumnName = "customersId")
  private Customers customer;

  
}
