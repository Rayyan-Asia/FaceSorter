package com.facesorter;

import com.facesorter.dto.*;
import com.facesorter.entity.AccountRole;
import org.junit.jupiter.api.Test;
import org.springframework.http.HttpMethod;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;

import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;

class StudioIntegrationTest extends AbstractIntegrationTest {

    /** Creates a studio (as admin) and returns its id, for tests that need a fresh studio. */
    private Long createStudio(String adminToken) {
        CreateStudioRequest req = CreateStudioRequest.builder()
                .name("Studio " + uniqueSuffix()).email("studio-" + uniqueSuffix() + "@example.com").build();
        return restTemplate.exchange("/api/admin/studios", HttpMethod.POST, authEntity(adminToken, req), StudioDto.class)
                .getBody().getId();
    }

    @Test
    void operatorAndAdminCanBothCreateEvents() {
        String adminToken = loginAsSeededAdmin();
        Long studioId = createStudio(adminToken);
        String operatorToken = registerAndLogin(AccountRole.STUDIO_OPERATOR, studioId);

        CreateEventRequest asOperator = CreateEventRequest.builder()
                .studioId(studioId).name("Operator Event").eventDate(java.time.LocalDate.now()).build();
        ResponseEntity<EventDto> r1 = restTemplate.exchange(
                "/api/studio/events", HttpMethod.POST, authEntity(operatorToken, asOperator), EventDto.class);
        assertThat(r1.getStatusCode()).isEqualTo(HttpStatus.OK);

        CreateEventRequest asAdmin = CreateEventRequest.builder()
                .studioId(studioId).name("Admin Event").eventDate(java.time.LocalDate.now()).build();
        ResponseEntity<EventDto> r2 = restTemplate.exchange(
                "/api/studio/events", HttpMethod.POST, authEntity(adminToken, asAdmin), EventDto.class);
        assertThat(r2.getStatusCode()).isEqualTo(HttpStatus.OK);
    }

    @Test
    void createEventMissingStudioIdReturns400() {
        String operatorToken = registerAndLogin(AccountRole.STUDIO_OPERATOR, null);
        CreateEventRequest missingStudio = CreateEventRequest.builder().name("No Studio").build();

        ResponseEntity<String> response = restTemplate.exchange(
                "/api/studio/events", HttpMethod.POST, authEntity(operatorToken, missingStudio), String.class);

        assertThat(response.getStatusCode()).isEqualTo(HttpStatus.BAD_REQUEST);
    }

    @Test
    void createEventWithNonexistentStudioReturns404() {
        String operatorToken = registerAndLogin(AccountRole.STUDIO_OPERATOR, null);
        CreateEventRequest badStudio = CreateEventRequest.builder().studioId(999_999L).name("Ghost").build();

        ResponseEntity<String> response = restTemplate.exchange(
                "/api/studio/events", HttpMethod.POST, authEntity(operatorToken, badStudio), String.class);

        assertThat(response.getStatusCode()).isEqualTo(HttpStatus.NOT_FOUND);
    }

    @Test
    void getEventByIdReturns404ForUnknownEvent() {
        String operatorToken = registerAndLogin(AccountRole.STUDIO_OPERATOR, null);
        ResponseEntity<String> response = restTemplate.exchange(
                "/api/studio/events/999999", HttpMethod.GET, authEntity(operatorToken), String.class);
        assertThat(response.getStatusCode()).isEqualTo(HttpStatus.NOT_FOUND);
    }

    private EventDto createEvent(String token, Long studioId) {
        CreateEventRequest req = CreateEventRequest.builder()
                .studioId(studioId).name("Event " + uniqueSuffix()).eventDate(java.time.LocalDate.now()).build();
        return restTemplate.exchange("/api/studio/events", HttpMethod.POST, authEntity(token, req), EventDto.class)
                .getBody();
    }

    @Test
    void registeringPhotosIncreasesEventPhotoCount() {
        String adminToken = loginAsSeededAdmin();
        Long studioId = createStudio(adminToken);
        String operatorToken = registerAndLogin(AccountRole.STUDIO_OPERATOR, studioId);
        EventDto event = createEvent(operatorToken, studioId);

        RegisterPhotosRequest.PhotoEntry entry = RegisterPhotosRequest.PhotoEntry.builder()
                .filename("a.jpg").localPath("/local/a.jpg").url("http://device/a.jpg").fileHash("hash-" + uniqueSuffix())
                .build();
        RegisterPhotosRequest req = RegisterPhotosRequest.builder().eventId(event.getId()).photos(List.of(entry)).build();

        ResponseEntity<PhotoDto[]> response = restTemplate.exchange(
                "/api/studio/photos/register", HttpMethod.POST, authEntity(operatorToken, req), PhotoDto[].class);

        assertThat(response.getStatusCode()).isEqualTo(HttpStatus.OK);
        assertThat(response.getBody()).hasSize(1);

        ResponseEntity<EventDto> updated = restTemplate.exchange(
                "/api/studio/events/" + event.getId(), HttpMethod.GET, authEntity(operatorToken), EventDto.class);
        assertThat(updated.getBody().getTotalPhotos()).isEqualTo(1);
        assertThat(updated.getBody().getProcessedPhotos()).isEqualTo(0);
    }

    @Test
    void registeringSamePhotoHashTwiceIsDeduplicated() {
        String adminToken = loginAsSeededAdmin();
        Long studioId = createStudio(adminToken);
        String operatorToken = registerAndLogin(AccountRole.STUDIO_OPERATOR, studioId);
        EventDto event = createEvent(operatorToken, studioId);

        String sharedHash = "hash-" + uniqueSuffix();
        RegisterPhotosRequest.PhotoEntry entry = RegisterPhotosRequest.PhotoEntry.builder()
                .filename("dup.jpg").localPath("/local/dup.jpg").url("http://device/dup.jpg").fileHash(sharedHash).build();
        RegisterPhotosRequest req = RegisterPhotosRequest.builder().eventId(event.getId()).photos(List.of(entry)).build();

        restTemplate.exchange("/api/studio/photos/register", HttpMethod.POST, authEntity(operatorToken, req), PhotoDto[].class);
        ResponseEntity<PhotoDto[]> secondAttempt = restTemplate.exchange(
                "/api/studio/photos/register", HttpMethod.POST, authEntity(operatorToken, req), PhotoDto[].class);

        // The duplicate-hash entry is silently filtered out — second call inserts nothing new.
        assertThat(secondAttempt.getBody()).isEmpty();

        ResponseEntity<PhotoDto[]> allPhotos = restTemplate.exchange(
                "/api/studio/events/" + event.getId() + "/photos", HttpMethod.GET, authEntity(operatorToken), PhotoDto[].class);
        assertThat(allPhotos.getBody()).hasSize(1);
    }

    @Test
    void unprocessedPhotosListShrinksAfterEmbeddingSubmission() {
        String adminToken = loginAsSeededAdmin();
        Long studioId = createStudio(adminToken);
        String operatorToken = registerAndLogin(AccountRole.STUDIO_OPERATOR, studioId);
        EventDto event = createEvent(operatorToken, studioId);

        RegisterPhotosRequest.PhotoEntry entry = RegisterPhotosRequest.PhotoEntry.builder()
                .filename("proc.jpg").localPath("/local/proc.jpg").url("http://device/proc.jpg")
                .fileHash("hash-" + uniqueSuffix()).build();
        PhotoDto photo = restTemplate.exchange("/api/studio/photos/register", HttpMethod.POST,
                authEntity(operatorToken, RegisterPhotosRequest.builder().eventId(event.getId()).photos(List.of(entry)).build()),
                PhotoDto[].class).getBody()[0];

        ResponseEntity<PhotoDto[]> beforeProcessing = restTemplate.exchange(
                "/api/studio/events/" + event.getId() + "/photos/unprocessed", HttpMethod.GET, authEntity(operatorToken), PhotoDto[].class);
        assertThat(beforeProcessing.getBody()).hasSize(1);

        SubmitEmbeddingsRequest submit = SubmitEmbeddingsRequest.builder()
                .photoId(photo.getId()).eventId(event.getId()).embeddings(List.of(identityVector(1))).build();
        restTemplate.postForEntity("/api/faces/embeddings", submit, Void.class);

        ResponseEntity<PhotoDto[]> afterProcessing = restTemplate.exchange(
                "/api/studio/events/" + event.getId() + "/photos/unprocessed", HttpMethod.GET, authEntity(operatorToken), PhotoDto[].class);
        assertThat(afterProcessing.getBody()).isEmpty();
    }

    @Test
    void createOrderDefaultsToEventStudioWhenStudioIdOmitted() {
        String adminToken = loginAsSeededAdmin();
        Long studioId = createStudio(adminToken);
        String operatorToken = registerAndLogin(AccountRole.STUDIO_OPERATOR, studioId);
        EventDto event = createEvent(operatorToken, studioId);

        CreateOrderRequest req = CreateOrderRequest.builder().eventId(event.getId()).build();
        ResponseEntity<OrderDto> response = restTemplate.exchange(
                "/api/studio/orders", HttpMethod.POST, authEntity(operatorToken, req), OrderDto.class);

        assertThat(response.getStatusCode()).isEqualTo(HttpStatus.OK);
        assertThat(response.getBody().getStudioId()).isEqualTo(studioId);
        assertThat(response.getBody().getStatus()).isEqualTo("CREATED");
    }

    @Test
    void addOrderItemsWithUnknownPhotoIdReturns404() {
        String adminToken = loginAsSeededAdmin();
        Long studioId = createStudio(adminToken);
        String operatorToken = registerAndLogin(AccountRole.STUDIO_OPERATOR, studioId);
        EventDto event = createEvent(operatorToken, studioId);
        OrderDto order = restTemplate.exchange("/api/studio/orders", HttpMethod.POST,
                authEntity(operatorToken, CreateOrderRequest.builder().eventId(event.getId()).build()), OrderDto.class).getBody();

        AddOrderItemsRequest req = AddOrderItemsRequest.builder().photoIds(List.of(999_999L)).build();
        ResponseEntity<String> response = restTemplate.exchange(
                "/api/studio/orders/" + order.getId() + "/items", HttpMethod.POST, authEntity(operatorToken, req), String.class);

        assertThat(response.getStatusCode()).isEqualTo(HttpStatus.NOT_FOUND);
    }

    @Test
    void updateOrderStatusPersists() {
        String adminToken = loginAsSeededAdmin();
        Long studioId = createStudio(adminToken);
        String operatorToken = registerAndLogin(AccountRole.STUDIO_OPERATOR, studioId);
        EventDto event = createEvent(operatorToken, studioId);
        OrderDto order = restTemplate.exchange("/api/studio/orders", HttpMethod.POST,
                authEntity(operatorToken, CreateOrderRequest.builder().eventId(event.getId()).build()), OrderDto.class).getBody();

        ResponseEntity<OrderDto> response = restTemplate.exchange(
                "/api/studio/orders/" + order.getId() + "/status?status=PROCESSING",
                HttpMethod.PUT, authEntity(operatorToken), OrderDto.class);

        assertThat(response.getBody().getStatus()).isEqualTo("PROCESSING");
    }
}
