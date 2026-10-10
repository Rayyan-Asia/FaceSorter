package com.facesorter;

import com.facesorter.dto.AuthResponse;
import com.facesorter.dto.LoginRequest;
import com.facesorter.dto.RegisterRequest;
import com.facesorter.entity.AccountRole;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.web.client.TestRestTemplate;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.test.context.DynamicPropertyRegistry;
import org.springframework.test.context.DynamicPropertySource;
import org.testcontainers.containers.PostgreSQLContainer;
import org.testcontainers.utility.DockerImageName;

import java.util.Random;
import java.util.UUID;

/**
 * Base class for all backend integration tests.
 *
 * Boots the full Spring context on a random port against a real Postgres+pgvector
 * instance (Testcontainers), so tests exercise the actual security filter chain,
 * Flyway migrations, and native pgvector similarity queries — not mocks.
 *
 * The container is started once per JVM and shared across all subclasses.
 */
@SpringBootTest(
        webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT,
        // Tests share one client IP and log in hundreds of times; the limiter has its own unit test.
        properties = {
                "facesorter.rate-limit.login-per-minute=1000000",
                "facesorter.rate-limit.customer-per-minute=1000000"
        })
public abstract class AbstractIntegrationTest {

    static final PostgreSQLContainer<?> POSTGRES =
            new PostgreSQLContainer<>(
                    DockerImageName.parse("pgvector/pgvector:pg16").asCompatibleSubstituteFor("postgres"))
                    .withDatabaseName("facesorter")
                    .withUsername("facesorter")
                    .withPassword("facesorter");

    static {
        POSTGRES.start();
    }

    @DynamicPropertySource
    static void datasourceProperties(DynamicPropertyRegistry registry) {
        registry.add("spring.datasource.url", POSTGRES::getJdbcUrl);
        registry.add("spring.datasource.username", POSTGRES::getUsername);
        registry.add("spring.datasource.password", POSTGRES::getPassword);
    }

    @Autowired
    protected TestRestTemplate restTemplate;

    protected static String uniqueSuffix() {
        return UUID.randomUUID().toString().substring(0, 8);
    }

    /**
     * Creates an account the way production does: an authenticated admin calls /register.
     * A studio operator always needs a studio, so one is created when none is supplied.
     */
    protected String registerAndLogin(AccountRole role, Long studioId) {
        String adminToken = loginAsSeededAdmin();
        Long effectiveStudioId = studioId;
        if (role == AccountRole.STUDIO_OPERATOR && effectiveStudioId == null) {
            effectiveStudioId = createStudioAsAdmin(adminToken);
        }
        String email = "test-" + uniqueSuffix() + "@example.com";
        RegisterRequest request = RegisterRequest.builder()
                .email(email)
                .password("Password123!")
                .role(role)
                .studioId(effectiveStudioId)
                .build();
        AuthResponse response = restTemplate.exchange("/api/auth/register", org.springframework.http.HttpMethod.POST,
                authEntity(adminToken, request), AuthResponse.class).getBody();
        return response.getToken();
    }

    protected Long createStudioAsAdmin(String adminToken) {
        com.facesorter.dto.CreateStudioRequest req = com.facesorter.dto.CreateStudioRequest.builder()
                .name("Studio " + uniqueSuffix()).email("studio-" + uniqueSuffix() + "@example.com").build();
        return restTemplate.exchange("/api/admin/studios", org.springframework.http.HttpMethod.POST,
                authEntity(adminToken, req), com.facesorter.dto.StudioDto.class).getBody().getId();
    }

    protected String loginAsSeededAdmin() {
        LoginRequest request = LoginRequest.builder()
                .email("admin@facesorter.com")
                .password("Admin1234!")
                .build();
        AuthResponse response = restTemplate.postForObject("/api/auth/login", request, AuthResponse.class);
        return response.getToken();
    }

    protected HttpEntity<Void> authEntity(String token) {
        HttpHeaders headers = new HttpHeaders();
        headers.setBearerAuth(token);
        return new HttpEntity<>(headers);
    }

    protected <T> HttpEntity<T> authEntity(String token, T body) {
        HttpHeaders headers = new HttpHeaders();
        headers.setBearerAuth(token);
        headers.setContentType(org.springframework.http.MediaType.APPLICATION_JSON);
        return new HttpEntity<>(body, headers);
    }

    /**
     * Deterministic pseudo-random 512-dim "identity" vector for a given seed.
     * Two different seeds are, with overwhelming probability, nearly orthogonal
     * (cosine distance ~1.0) — reliably "not similar" under the app's 0.65 threshold.
     */
    protected static float[] identityVector(long seed) {
        Random random = new Random(seed);
        float[] vector = new float[512];
        for (int i = 0; i < vector.length; i++) {
            vector[i] = (float) random.nextGaussian();
        }
        return vector;
    }

    /**
     * A small perturbation of a base identity vector — simulates a different photo
     * of the same person. Cosine distance stays well under 0.65 ("similar").
     */
    protected static float[] nearbyVariant(float[] base, long noiseSeed) {
        Random random = new Random(noiseSeed);
        float[] variant = new float[base.length];
        for (int i = 0; i < base.length; i++) {
            variant[i] = base[i] + 0.05f * (float) random.nextGaussian();
        }
        return variant;
    }
}
