package com.facesorter.security;

import com.facesorter.entity.Account;
import io.jsonwebtoken.Claims;
import io.jsonwebtoken.Jwts;
import io.jsonwebtoken.security.Keys;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

import javax.crypto.SecretKey;
import java.nio.charset.StandardCharsets;
import java.util.Date;

@Component
public class JwtUtil {

    @Value("${facesorter.jwt.secret}")
    private String secret;

    @Value("${facesorter.jwt.expiry-hours:24}")
    private long expiryHours;

    private static final String DEV_DEFAULT_SECRET = "facesorter-dev-secret-key-change-in-production-2026";

    @Value("${spring.profiles.active:}")
    private String activeProfiles;

    private static final org.slf4j.Logger log = org.slf4j.LoggerFactory.getLogger(JwtUtil.class);

    /** Refuse to run a production profile with a guessable/default signing key (forgeable admin tokens). */
    @jakarta.annotation.PostConstruct
    void validateSecret() {
        if (secret == null || secret.getBytes(StandardCharsets.UTF_8).length < 32) {
            throw new IllegalStateException("JWT secret must be at least 32 bytes");
        }
        if (DEV_DEFAULT_SECRET.equals(secret)) {
            if (activeProfiles.contains("prod")) {
                throw new IllegalStateException("JWT_SECRET must be set to a non-default value in the prod profile");
            }
            log.warn("Using the built-in development JWT secret. Set JWT_SECRET before deploying.");
        }
    }

    private SecretKey key() {
        return Keys.hmacShaKeyFor(secret.getBytes(StandardCharsets.UTF_8));
    }

    public String generate(Account account) {
        return Jwts.builder()
                .subject(account.getEmail())
                .claim("role", account.getRole().name())
                .claim("studioId", account.getStudio() != null ? account.getStudio().getId() : null)
                .issuedAt(new Date())
                .expiration(new Date(System.currentTimeMillis() + expiryHours * 3_600_000L))
                .signWith(key())
                .compact();
    }

    public String extractEmail(String token) {
        return parseClaims(token).getSubject();
    }

    public boolean isValid(String token) {
        try {
            parseClaims(token);
            return true;
        } catch (Exception e) {
            return false;
        }
    }

    private Claims parseClaims(String token) {
        return Jwts.parser()
                .verifyWith(key())
                .build()
                .parseSignedClaims(token)
                .getPayload();
    }
}
