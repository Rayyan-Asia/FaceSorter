package com.facesorter.service;

import com.facesorter.dto.AuthResponse;
import com.facesorter.dto.LoginRequest;
import com.facesorter.dto.RegisterRequest;
import com.facesorter.entity.Account;
import com.facesorter.entity.AccountRole;
import com.facesorter.entity.Studio;
import com.facesorter.repository.AccountRepository;
import com.facesorter.repository.StudioRepository;
import com.facesorter.security.JwtUtil;
import lombok.RequiredArgsConstructor;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.security.access.AccessDeniedException;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

@Service
@RequiredArgsConstructor
public class AuthService {

    private final AccountRepository accountRepository;
    private final StudioRepository studioRepository;
    private final PasswordEncoder passwordEncoder;
    private final JwtUtil jwtUtil;

    @Transactional
    public AuthResponse register(RegisterRequest request) {
        if (accountRepository.findByEmail(request.getEmail()).isPresent()) {
            throw new IllegalArgumentException("Email already registered");
        }

        Studio studio = null;
        if (request.getStudioId() != null) {
            studio = studioRepository.findById(request.getStudioId())
                    .orElseThrow(() -> new IllegalArgumentException("Unknown studio"));
        }
        // An operator without a studio could never be scoped to anything; refuse to create one.
        if (request.getRole() == AccountRole.STUDIO_OPERATOR && studio == null) {
            throw new IllegalArgumentException("studioId is required for studio operators");
        }

        Account account = Account.builder()
                .email(request.getEmail())
                .passwordHash(passwordEncoder.encode(request.getPassword()))
                .role(request.getRole())
                .studio(studio)
                .build();

        accountRepository.save(account);
        return toResponse(account, jwtUtil.generate(account));
    }

    @Transactional(readOnly = true)
    public AuthResponse login(LoginRequest request) {
        Account account = accountRepository.findByEmail(request.getEmail())
                .orElseThrow(() -> new IllegalArgumentException("Invalid credentials"));

        if (!passwordEncoder.matches(request.getPassword(), account.getPasswordHash())) {
            throw new IllegalArgumentException("Invalid credentials");
        }

        // Subscription gating (NFR-4): a deactivated studio's operators cannot start a session.
        if (account.getRole() == AccountRole.STUDIO_OPERATOR && account.getStudio() != null
                && !Boolean.TRUE.equals(account.getStudio().getActive())) {
            throw new AccessDeniedException("Studio is deactivated");
        }

        return toResponse(account, jwtUtil.generate(account));
    }

    private AuthResponse toResponse(Account account, String token) {
        return AuthResponse.builder()
                .token(token)
                .email(account.getEmail())
                .role(account.getRole().name())
                .studioId(account.getStudio() != null ? account.getStudio().getId() : null)
                .build();
    }
}
