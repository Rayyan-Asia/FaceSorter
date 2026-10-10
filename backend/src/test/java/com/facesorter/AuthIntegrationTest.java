package com.facesorter;

import com.facesorter.dto.AuthResponse;
import com.facesorter.dto.LoginRequest;
import com.facesorter.dto.RegisterRequest;
import com.facesorter.entity.AccountRole;
import org.junit.jupiter.api.Test;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;

import static org.assertj.core.api.Assertions.assertThat;

class AuthIntegrationTest extends AbstractIntegrationTest {

    @Test
    void registerThenLoginReturnsWorkingToken() {
        String email = "test-" + uniqueSuffix() + "@example.com";
        RegisterRequest register = RegisterRequest.builder()
                .email(email).password("Password123!").role(AccountRole.STUDIO_OPERATOR).build();

        ResponseEntity<AuthResponse> registerResponse =
                restTemplate.postForEntity("/api/auth/register", register, AuthResponse.class);

        assertThat(registerResponse.getStatusCode()).isEqualTo(HttpStatus.OK);
        assertThat(registerResponse.getBody().getToken()).isNotBlank();
        assertThat(registerResponse.getBody().getEmail()).isEqualTo(email);
        assertThat(registerResponse.getBody().getRole()).isEqualTo("STUDIO_OPERATOR");

        LoginRequest login = LoginRequest.builder().email(email).password("Password123!").build();
        ResponseEntity<AuthResponse> loginResponse =
                restTemplate.postForEntity("/api/auth/login", login, AuthResponse.class);

        assertThat(loginResponse.getStatusCode()).isEqualTo(HttpStatus.OK);
        assertThat(loginResponse.getBody().getToken()).isNotBlank();
    }

    @Test
    void registerWithDuplicateEmailReturns409() {
        String email = "test-" + uniqueSuffix() + "@example.com";
        RegisterRequest register = RegisterRequest.builder()
                .email(email).password("Password123!").role(AccountRole.ADMIN).build();

        restTemplate.postForEntity("/api/auth/register", register, AuthResponse.class);
        ResponseEntity<AuthResponse> second =
                restTemplate.postForEntity("/api/auth/register", register, AuthResponse.class);

        assertThat(second.getStatusCode()).isEqualTo(HttpStatus.CONFLICT);
    }

    @Test
    void registerWithShortPasswordReturns400NotOpaque403() {
        RegisterRequest register = RegisterRequest.builder()
                .email("test-" + uniqueSuffix() + "@example.com")
                .password("short")
                .role(AccountRole.ADMIN)
                .build();

        ResponseEntity<String> response =
                restTemplate.postForEntity("/api/auth/register", register, String.class);

        // Regression test: validation failures must surface as 400, not be swallowed
        // as an opaque 403 by the security filter chain's handling of the /error forward.
        assertThat(response.getStatusCode()).isEqualTo(HttpStatus.BAD_REQUEST);
    }

    @Test
    void loginWithWrongPasswordReturns401() {
        String email = "test-" + uniqueSuffix() + "@example.com";
        restTemplate.postForEntity("/api/auth/register",
                RegisterRequest.builder().email(email).password("Password123!").role(AccountRole.ADMIN).build(),
                AuthResponse.class);

        ResponseEntity<AuthResponse> response = restTemplate.postForEntity("/api/auth/login",
                LoginRequest.builder().email(email).password("WrongPassword!").build(), AuthResponse.class);

        assertThat(response.getStatusCode()).isEqualTo(HttpStatus.UNAUTHORIZED);
    }

    @Test
    void loginWithUnknownEmailReturns401() {
        ResponseEntity<AuthResponse> response = restTemplate.postForEntity("/api/auth/login",
                LoginRequest.builder().email("nobody-" + uniqueSuffix() + "@example.com").password("whatever1").build(),
                AuthResponse.class);

        assertThat(response.getStatusCode()).isEqualTo(HttpStatus.UNAUTHORIZED);
    }

    @Test
    void meWithoutTokenReturns401() {
        ResponseEntity<AuthResponse> response = restTemplate.getForEntity("/api/auth/me", AuthResponse.class);
        assertThat(response.getStatusCode()).isEqualTo(HttpStatus.UNAUTHORIZED);
    }

    @Test
    void meWithValidTokenReturnsAccountInfo() {
        String token = registerAndLogin(AccountRole.ADMIN, null);

        ResponseEntity<AuthResponse> response = restTemplate.exchange(
                "/api/auth/me", org.springframework.http.HttpMethod.GET, authEntity(token), AuthResponse.class);

        assertThat(response.getStatusCode()).isEqualTo(HttpStatus.OK);
        assertThat(response.getBody().getRole()).isEqualTo("ADMIN");
    }

    @Test
    void protectedEndpointWithInvalidTokenReturns401() {
        ResponseEntity<String> response = restTemplate.exchange(
                "/api/studio/events?studioId=1", org.springframework.http.HttpMethod.GET,
                authEntity("not.a.valid-token"), String.class);
        assertThat(response.getStatusCode()).isEqualTo(HttpStatus.UNAUTHORIZED);
    }

    @Test
    void protectedEndpointWithoutTokenReturns401() {
        ResponseEntity<String> response = restTemplate.getForEntity("/api/studio/events?studioId=1", String.class);
        assertThat(response.getStatusCode()).isEqualTo(HttpStatus.UNAUTHORIZED);
    }

    @Test
    void operatorOnAdminEndpointStillReturns403() {
        String token = registerAndLogin(AccountRole.STUDIO_OPERATOR, null);
        ResponseEntity<String> response = restTemplate.exchange(
                "/api/admin/studios", org.springframework.http.HttpMethod.GET, authEntity(token), String.class);
        assertThat(response.getStatusCode()).isEqualTo(HttpStatus.FORBIDDEN);
    }

    @Test
    void seededAdminAccountCanLogIn() {
        String token = loginAsSeededAdmin();
        assertThat(token).isNotBlank();
    }
}
