package com.facesorter.controller;

import org.springframework.http.ResponseEntity;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.security.oauth2.core.user.OAuth2User;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.Map;

@RestController
@RequestMapping("/api/auth")
public class AuthController {

    @GetMapping("/me")
    public ResponseEntity<Map<String, Object>> getCurrentUser(
            @AuthenticationPrincipal OAuth2User principal) {
        if (principal == null) {
            return ResponseEntity.ok(Map.of("authenticated", false));
        }

        return ResponseEntity.ok(Map.of(
                "authenticated", true,
                "name", principal.getAttribute("name"),
                "email", principal.getAttribute("email"),
                "sub", principal.getAttribute("sub")
        ));
    }

    @GetMapping("/login/success")
    public ResponseEntity<Map<String, String>> loginSuccess() {
        return ResponseEntity.ok(Map.of("status", "ok", "message", "Login successful"));
    }
}
