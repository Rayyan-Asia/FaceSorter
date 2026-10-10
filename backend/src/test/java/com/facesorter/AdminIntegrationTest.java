package com.facesorter;

import com.facesorter.dto.*;
import com.facesorter.entity.AccountRole;
import org.junit.jupiter.api.Test;
import org.springframework.http.HttpMethod;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;

import java.math.BigDecimal;
import java.time.LocalDate;

import static org.assertj.core.api.Assertions.assertThat;

class AdminIntegrationTest extends AbstractIntegrationTest {

    @Test
    void nonAdminIsBlockedFromAdminEndpoints() {
        String operatorToken = registerAndLogin(AccountRole.STUDIO_OPERATOR, null);

        ResponseEntity<String> response = restTemplate.exchange(
                "/api/admin/studios", HttpMethod.GET, authEntity(operatorToken), String.class);

        assertThat(response.getStatusCode()).isEqualTo(HttpStatus.FORBIDDEN);
    }

    @Test
    void unauthenticatedIsBlockedFromAdminEndpoints() {
        ResponseEntity<String> response = restTemplate.getForEntity("/api/admin/studios", String.class);
        assertThat(response.getStatusCode()).isEqualTo(HttpStatus.UNAUTHORIZED);
    }

    @Test
    void dashboardStatsReflectsCreatedStudio() {
        String adminToken = loginAsSeededAdmin();

        ResponseEntity<java.util.Map> before = restTemplate.exchange(
                "/api/admin/dashboard/stats", HttpMethod.GET, authEntity(adminToken), java.util.Map.class);
        long studiosBefore = ((Number) before.getBody().get("totalStudios")).longValue();

        CreateStudioRequest create = CreateStudioRequest.builder()
                .name("Stats Studio " + uniqueSuffix()).email("stats-" + uniqueSuffix() + "@studio.com").build();
        restTemplate.exchange("/api/admin/studios", HttpMethod.POST, authEntity(adminToken, create), StudioDto.class);

        ResponseEntity<java.util.Map> after = restTemplate.exchange(
                "/api/admin/dashboard/stats", HttpMethod.GET, authEntity(adminToken), java.util.Map.class);
        long studiosAfter = ((Number) after.getBody().get("totalStudios")).longValue();

        assertThat(studiosAfter).isEqualTo(studiosBefore + 1);
    }

    @Test
    void createStudioThenActivateDeactivateRoundTrips() {
        String adminToken = loginAsSeededAdmin();
        CreateStudioRequest create = CreateStudioRequest.builder()
                .name("Toggle Studio").email("toggle-" + uniqueSuffix() + "@studio.com").build();

        ResponseEntity<StudioDto> created = restTemplate.exchange(
                "/api/admin/studios", HttpMethod.POST, authEntity(adminToken, create), StudioDto.class);
        assertThat(created.getStatusCode()).isEqualTo(HttpStatus.OK);
        assertThat(created.getBody().getActive()).isTrue();
        Long studioId = created.getBody().getId();

        ResponseEntity<StudioDto> deactivated = restTemplate.exchange(
                "/api/admin/studios/" + studioId + "/deactivate", HttpMethod.PUT, authEntity(adminToken), StudioDto.class);
        assertThat(deactivated.getBody().getActive()).isFalse();

        ResponseEntity<StudioDto> reactivated = restTemplate.exchange(
                "/api/admin/studios/" + studioId + "/activate", HttpMethod.PUT, authEntity(adminToken), StudioDto.class);
        assertThat(reactivated.getBody().getActive()).isTrue();
    }

    @Test
    void createStudioWithDuplicateEmailReturns409NotOpaque500() {
        String adminToken = loginAsSeededAdmin();
        String email = "dupe-" + uniqueSuffix() + "@studio.com";
        CreateStudioRequest create = CreateStudioRequest.builder().name("Dupe A").email(email).build();

        restTemplate.exchange("/api/admin/studios", HttpMethod.POST, authEntity(adminToken, create), StudioDto.class);
        ResponseEntity<String> second = restTemplate.exchange(
                "/api/admin/studios", HttpMethod.POST,
                authEntity(adminToken, CreateStudioRequest.builder().name("Dupe B").email(email).build()),
                String.class);

        // Regression test: unique-constraint violations must surface as 409, not 500.
        assertThat(second.getStatusCode()).isEqualTo(HttpStatus.CONFLICT);
    }

    @Test
    void createUserThenAppearsInList() {
        String adminToken = loginAsSeededAdmin();
        CreateUserRequest create = CreateUserRequest.builder()
                .idNumber("ID-" + uniqueSuffix()).name("Jane Customer").email("jane@example.com").build();

        ResponseEntity<UserDto> created = restTemplate.exchange(
                "/api/admin/users", HttpMethod.POST, authEntity(adminToken, create), UserDto.class);
        assertThat(created.getStatusCode()).isEqualTo(HttpStatus.OK);

        ResponseEntity<UserDto[]> list = restTemplate.exchange(
                "/api/admin/users", HttpMethod.GET, authEntity(adminToken), UserDto[].class);
        assertThat(list.getBody()).extracting(UserDto::getId).contains(created.getBody().getId());
    }

    @Test
    void createSubscriptionForNonexistentStudioReturns404() {
        String adminToken = loginAsSeededAdmin();
        CreateSubscriptionRequest create = CreateSubscriptionRequest.builder()
                .studioId(999_999L).startDate(LocalDate.now()).endDate(LocalDate.now().plusYears(1))
                .paymentMethod("CASH").amountPaid(BigDecimal.TEN).build();

        ResponseEntity<String> response = restTemplate.exchange(
                "/api/admin/subscriptions", HttpMethod.POST, authEntity(adminToken, create), String.class);

        assertThat(response.getStatusCode()).isEqualTo(HttpStatus.NOT_FOUND);
    }

    @Test
    void createSubscriptionSucceedsForRealStudio() {
        String adminToken = loginAsSeededAdmin();
        CreateStudioRequest studioReq = CreateStudioRequest.builder()
                .name("Sub Studio").email("sub-" + uniqueSuffix() + "@studio.com").build();
        StudioDto studio = restTemplate.exchange(
                "/api/admin/studios", HttpMethod.POST, authEntity(adminToken, studioReq), StudioDto.class).getBody();

        CreateSubscriptionRequest subReq = CreateSubscriptionRequest.builder()
                .studioId(studio.getId()).startDate(LocalDate.now()).endDate(LocalDate.now().plusYears(1))
                .paymentMethod("CASH").amountPaid(new BigDecimal("250.00")).build();

        ResponseEntity<SubscriptionDto> response = restTemplate.exchange(
                "/api/admin/subscriptions", HttpMethod.POST, authEntity(adminToken, subReq), SubscriptionDto.class);

        assertThat(response.getStatusCode()).isEqualTo(HttpStatus.OK);
        assertThat(response.getBody().getStatus()).isEqualTo("active");
    }

    @Test
    void getOrderByIdReturns404ForUnknownOrder() {
        String adminToken = loginAsSeededAdmin();
        ResponseEntity<String> response = restTemplate.exchange(
                "/api/admin/orders/999999", HttpMethod.GET, authEntity(adminToken), String.class);
        assertThat(response.getStatusCode()).isEqualTo(HttpStatus.NOT_FOUND);
    }
}
