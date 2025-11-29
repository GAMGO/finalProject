package org.iclass.profile.service;

import org.iclass.profile.dto.UserProfileDto;
import org.iclass.profile.entity.UserProfileEntity;
import org.iclass.profile.repository.UserProfileRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

@Service
public class UserProfileService {

    private final UserProfileRepository userProfileRepository;

    @Autowired
    public UserProfileService(UserProfileRepository userProfileRepository) {
        this.userProfileRepository = userProfileRepository;
    }

    // TODO: 나중에 인증 붙으면 여기만 바꾸면 됨
    private Long getcustomerIdx() {
        return 1L;
    }

    public UserProfileDto getMyProfile() {
        Long customerIdx = getcustomerIdx();

        UserProfileEntity entity = userProfileRepository.findBycustomerIdx(customerIdx)
                .orElseGet(() -> {
                    UserProfileEntity e = new UserProfileEntity();
                    e.setCustomerIdx(customerIdx);
                    return userProfileRepository.save(e);
                });

        return toDto(entity);
    }

    public UserProfileDto saveMyProfile(UserProfileDto dto) {
        Long customerIdx = getcustomerIdx();

        UserProfileEntity entity = userProfileRepository.findBycustomerIdx(customerIdx)
                .orElseGet(() -> {
                    UserProfileEntity e = new UserProfileEntity();
                    e.setCustomerIdx(customerIdx);
                    return e;
                });

        entity.setNickname(dto.getNickname());
        entity.setIntro(dto.getIntro());
        entity.setAvatarUrl(dto.getAvatarUrl());
        entity.setFavoriteFood(dto.getFavoriteFood());
        entity.setLocation(dto.getLocation());

        UserProfileEntity saved = userProfileRepository.save(entity);
        return toDto(saved);
    }

    private UserProfileDto toDto(UserProfileEntity entity) {
        UserProfileDto dto = new UserProfileDto();
        dto.setIdx(entity.getIdx());
        dto.setNickname(entity.getNickname());
        dto.setIntro(entity.getIntro());
        dto.setAvatarUrl(entity.getAvatarUrl());
        dto.setFavoriteFood(entity.getFavoriteFood());
        dto.setLocation(entity.getLocation());
        return dto;
    }
}
