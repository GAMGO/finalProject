package com.finalproject.backend.dto;

import lombok.Data;

@Data
public class SignupRequest {
  private String customersId;
  private String password;
  private String name;
  private String birth;
  private Integer gender;
}