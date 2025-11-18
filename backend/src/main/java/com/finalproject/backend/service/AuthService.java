package com.finalproject.backend.service;

import org.springframework.stereotype.Service;

import com.finalproject.backend.dto.SignupRequest;
import com.finalproject.backend.entity.Customers;
import com.finalproject.backend.entity.UserProfile;
import com.finalproject.backend.repository.CustomersRepository;
import com.finalproject.backend.repository.UserProfileRepository;

import java.util.UUID;

import org.springframework.security.crypto.password.PasswordEncoder;
import lombok.RequiredArgsConstructor;

@Service
@RequiredArgsConstructor
public class AuthService {

  private final CustomersRepository customersRepository;
  private final UserProfileRepository profileRepository;
  private final PasswordEncoder passwordEncoder;

  public Customers findByEmail(String email) {
    return customersRepository.findByEmail(email).orElse(null);
  }

  public void signup(SignupRequest dto) {
    if (customersRepository.existsByCustomersId(dto.getCustomersId())) {
      throw new RuntimeException("이미 존재하는 아이디입니다.");
    }

    Customers customer = new Customers();
    customer.setCustomersId(dto.getCustomersId());
    customer.setPassword(passwordEncoder.encode(dto.getPassword()));
    customer.setEmailVerified(0);

    customersRepository.save(customer);

    UserProfile profile = new UserProfile();
    profile.setName(dto.getName());
    profile.setBirth(dto.getBirth());
    profile.setGender(dto.getGender());
    profile.setCustomer(customer);

    profileRepository.save(profile);
  }

  public Customers googleSignup(String email, String name, String picture) {
    Customers user = new Customers();
    user.setCustomersId(email);
    user.setPassword(UUID.randomUUID().toString()); // 랜덤 값
    user.setEmailVerified(1);
    user.setUserProfile(new UserProfile());

    return customersRepository.save(user);
  }

}
