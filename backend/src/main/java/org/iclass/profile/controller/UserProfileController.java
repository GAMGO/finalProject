package org.iclass.profile.controller;

import org.iclass.profile.dto.UserProfileDto;
import org.iclass.profile.service.UserProfileService;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/profile")
public class UserProfileController {

    private final UserProfileService userProfileService;

    public UserProfileController(UserProfileService userProfileService) {
        this.userProfileService = userProfileService;
    }

    @GetMapping
    public UserProfileDto getProfile() {
        // ApiResponse 안 쓰고 바로 DTO 리턴
        return userProfileService.getMyProfile();
    }

    @PostMapping
    public UserProfileDto saveProfile(@RequestBody UserProfileDto dto) {
        return userProfileService.saveMyProfile(dto);
    }
}
