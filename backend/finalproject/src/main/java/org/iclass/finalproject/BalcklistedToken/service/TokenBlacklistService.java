package org.iclass.finalproject.BalcklistedToken.service;

import java.time.LocalDateTime;

import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import org.iclass.finalproject.BalcklistedToken.entity.BlacklistedTokenEntity;
import org.iclass.finalproject.BalcklistedToken.repository.BlacklistedTokenRepository;

import lombok.RequiredArgsConstructor;

@RequiredArgsConstructor
@Service
public class TokenBlacklistService {

    private final BlacklistedTokenRepository repository;

    @Transactional(readOnly = true)
    public boolean isBlacklisted(String token) {
        return repository.existsByToken(token);
    }

    @Transactional
    public void blacklist(String token, Long customerIdx, LocalDateTime expiresAt, String reason) {
        if (repository.existsByToken(token)) return;
        BlacklistedTokenEntity e = BlacklistedTokenEntity.builder()
                .token(token)
                .customerIdx(customerIdx)
                .expiresAt(expiresAt)
                .createdAt(LocalDateTime.now())
                .reason(reason)
                .build();
        repository.save(e);
    }

    @Transactional
    public int purgeExpired() {
        return repository.deleteAllExpired(LocalDateTime.now());
    }
}
