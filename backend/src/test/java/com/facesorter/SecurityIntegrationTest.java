package com.facesorter;

import com.facesorter.dto.*;
import com.facesorter.entity.AccountRole;
import org.junit.jupiter.api.Test;
import org.springframework.http.HttpMethod;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;

import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * Authorization / abuse-resistance tests: tenant isolation between studios, the public customer
 * endpoints' limits, studio deactivation, and input whitelisting.
 */
class SecurityIntegrationTest extends AbstractIntegrationTest {

    /** A studio with one operator, one event, one registered+processed photo and one order. */
    private record Tenant(String token, Long studioId, Long eventId, Long photoId, Long orderId, float[] embedding) {}

    private Tenant newTenant(int identitySeed) {
        String adminToken = loginAsSeededAdmin();
        Long studioId = createStudioAsAdmin(adminToken);
        String token = registerAndLogin(AccountRole.STUDIO_OPERATOR, studioId);

        Long eventId = restTemplate.exchange("/api/studio/events", HttpMethod.POST, authEntity(token,
                CreateEventRequest.builder().studioId(studioId).name("Event " + uniqueSuffix()).build()),
                EventDto.class).getBody().getId();

        RegisterPhotosRequest.PhotoEntry entry = RegisterPhotosRequest.PhotoEntry.builder()
                .filename("p-" + uniqueSuffix() + ".jpg").localPath("/private/disk/path.jpg")
                .url("http://device/p.jpg").fileHash("h-" + uniqueSuffix()).build();
        Long photoId = restTemplate.exchange("/api/studio/photos/register", HttpMethod.POST, authEntity(token,
                RegisterPhotosRequest.builder().eventId(eventId).photos(List.of(entry)).build()),
                PhotoDto[].class).getBody()[0].getId();

        float[] embedding = identityVector(identitySeed);
        restTemplate.exchange("/api/faces/embeddings", HttpMethod.POST, authEntity(token,
                SubmitEmbeddingsRequest.builder().photoId(photoId).eventId(eventId)
                        .embeddings(List.of(embedding)).build()), Void.class);

        Long orderId = restTemplate.exchange("/api/studio/orders", HttpMethod.POST, authEntity(token,
                CreateOrderRequest.builder().eventId(eventId).build()), OrderDto.class).getBody().getId();
        return new Tenant(token, studioId, eventId, photoId, orderId, embedding);
    }

    private ResponseEntity<String> get(String token, String path) {
        return restTemplate.exchange(path, HttpMethod.GET, authEntity(token), String.class);
    }

    // --- Tenant isolation ---

    @Test
    void operatorCannotReadAnotherStudiosData() {
        Tenant a = newTenant(1100);
        Tenant b = newTenant(1200);

        assertThat(get(a.token(), "/api/studio/events?studioId=" + b.studioId()).getStatusCode()).isEqualTo(HttpStatus.FORBIDDEN);
        assertThat(get(a.token(), "/api/studio/events/" + b.eventId()).getStatusCode()).isEqualTo(HttpStatus.FORBIDDEN);
        assertThat(get(a.token(), "/api/studio/events/" + b.eventId() + "/photos").getStatusCode()).isEqualTo(HttpStatus.FORBIDDEN);
        assertThat(get(a.token(), "/api/studio/orders?studioId=" + b.studioId()).getStatusCode()).isEqualTo(HttpStatus.FORBIDDEN);
        assertThat(get(a.token(), "/api/studio/orders/" + b.orderId()).getStatusCode()).isEqualTo(HttpStatus.FORBIDDEN);
    }

    @Test
    void operatorCannotModifyAnotherStudiosData() {
        Tenant a = newTenant(1300);
        Tenant b = newTenant(1400);

        ResponseEntity<String> createOrder = restTemplate.exchange("/api/studio/orders", HttpMethod.POST,
                authEntity(a.token(), CreateOrderRequest.builder().eventId(b.eventId()).build()), String.class);
        assertThat(createOrder.getStatusCode()).isEqualTo(HttpStatus.FORBIDDEN);

        ResponseEntity<String> status = restTemplate.exchange(
                "/api/studio/orders/" + b.orderId() + "/status?status=FULFILLED", HttpMethod.PUT,
                authEntity(a.token()), String.class);
        assertThat(status.getStatusCode()).isEqualTo(HttpStatus.FORBIDDEN);

        ResponseEntity<String> items = restTemplate.exchange("/api/studio/orders/" + b.orderId() + "/items",
                HttpMethod.POST, authEntity(a.token(),
                        AddOrderItemsRequest.builder().photoIds(List.of(b.photoId())).build()), String.class);
        assertThat(items.getStatusCode()).isEqualTo(HttpStatus.FORBIDDEN);

        RegisterPhotosRequest.PhotoEntry entry = RegisterPhotosRequest.PhotoEntry.builder()
                .filename("x.jpg").localPath("/p/x.jpg").url("http://device/x.jpg").fileHash("h-" + uniqueSuffix()).build();
        ResponseEntity<String> register = restTemplate.exchange("/api/studio/photos/register", HttpMethod.POST,
                authEntity(a.token(), RegisterPhotosRequest.builder().eventId(b.eventId()).photos(List.of(entry)).build()),
                String.class);
        assertThat(register.getStatusCode()).isEqualTo(HttpStatus.FORBIDDEN);
    }

    @Test
    void operatorCannotSearchOrPoisonAnotherStudiosFaces() {
        Tenant a = newTenant(1500);
        Tenant b = newTenant(1600);

        ResponseEntity<String> search = restTemplate.exchange("/api/faces/search", HttpMethod.POST, authEntity(a.token(),
                FaceSearchRequest.builder().eventId(b.eventId()).embedding(b.embedding()).build()), String.class);
        assertThat(search.getStatusCode()).isEqualTo(HttpStatus.FORBIDDEN);

        ResponseEntity<String> submit = restTemplate.exchange("/api/faces/embeddings", HttpMethod.POST,
                authEntity(a.token(), SubmitEmbeddingsRequest.builder().photoId(b.photoId()).eventId(b.eventId())
                        .embeddings(List.of(identityVector(1))).build()), String.class);
        assertThat(submit.getStatusCode()).isEqualTo(HttpStatus.FORBIDDEN);
    }

    @Test
    void embeddingsCannotBeFiledUnderADifferentEventThanThePhotoBelongsTo() {
        Tenant a = newTenant(1700);
        // A second event in the SAME studio, so tenancy passes and only the photo/event check can stop it.
        Long otherEvent = restTemplate.exchange("/api/studio/events", HttpMethod.POST, authEntity(a.token(),
                CreateEventRequest.builder().studioId(a.studioId()).name("Other " + uniqueSuffix()).build()),
                EventDto.class).getBody().getId();
        RegisterPhotosRequest.PhotoEntry entry = RegisterPhotosRequest.PhotoEntry.builder()
                .filename("y.jpg").localPath("/p/y.jpg").url("http://device/y.jpg").fileHash("h-" + uniqueSuffix()).build();
        Long photoInOther = restTemplate.exchange("/api/studio/photos/register", HttpMethod.POST, authEntity(a.token(),
                RegisterPhotosRequest.builder().eventId(otherEvent).photos(List.of(entry)).build()),
                PhotoDto[].class).getBody()[0].getId();

        ResponseEntity<String> response = restTemplate.exchange("/api/faces/embeddings", HttpMethod.POST,
                authEntity(a.token(), SubmitEmbeddingsRequest.builder().photoId(photoInOther).eventId(a.eventId())
                        .embeddings(List.of(identityVector(2))).build()), String.class);

        assertThat(response.getStatusCode()).isEqualTo(HttpStatus.BAD_REQUEST);
    }

    @Test
    void faceEndpointsRequireAuthentication() {
        FaceSearchRequest search = FaceSearchRequest.builder().eventId(1L).embedding(identityVector(3)).build();
        assertThat(restTemplate.postForEntity("/api/faces/search", search, String.class).getStatusCode())
                .isEqualTo(HttpStatus.UNAUTHORIZED);
        assertThat(restTemplate.postForEntity("/api/faces/embeddings",
                SubmitEmbeddingsRequest.builder().photoId(1L).eventId(1L).embeddings(List.of(identityVector(3))).build(),
                String.class).getStatusCode()).isEqualTo(HttpStatus.UNAUTHORIZED);
    }

    @Test
    void adminIsNotRestrictedToOneStudio() {
        Tenant a = newTenant(1800);
        String adminToken = loginAsSeededAdmin();
        assertThat(get(adminToken, "/api/studio/events/" + a.eventId()).getStatusCode()).isEqualTo(HttpStatus.OK);
        assertThat(get(adminToken, "/api/studio/orders/" + a.orderId()).getStatusCode()).isEqualTo(HttpStatus.OK);
    }

    // --- Customer (public, order-id based) endpoints ---

    @Test
    void customerCannotResolveEmbeddingsFromAnotherEvent() {
        Tenant a = newTenant(1900);
        Tenant b = newTenant(2000);

        // Find B's embedding id via the (public) step-1 search on B's own order.
        EmbeddingSearchResult step1 = restTemplate.postForEntity("/api/customer/orders/" + b.orderId() + "/search",
                FaceSearchRequest.builder().embedding(b.embedding()).build(), EmbeddingSearchResult.class).getBody();
        Long bEmbeddingId = step1.getCandidates().get(0).getEmbeddingId();

        // ...then try to use it through A's order.
        FaceSearchResult leaked = restTemplate.postForEntity("/api/customer/orders/" + a.orderId() + "/photos",
                PhotosByEmbeddingsRequest.builder().embeddingIds(List.of(bEmbeddingId)).build(),
                FaceSearchResult.class).getBody();

        assertThat(leaked.getMatchedPhotos()).isEmpty();
    }

    @Test
    void customerResultsDoNotExposeStudioFilesystemPaths() {
        Tenant a = newTenant(2100);
        EmbeddingSearchResult step1 = restTemplate.postForEntity("/api/customer/orders/" + a.orderId() + "/search",
                FaceSearchRequest.builder().embedding(a.embedding()).build(), EmbeddingSearchResult.class).getBody();
        Long embeddingId = step1.getCandidates().get(0).getEmbeddingId();

        FaceSearchResult result = restTemplate.postForEntity("/api/customer/orders/" + a.orderId() + "/photos",
                PhotosByEmbeddingsRequest.builder().embeddingIds(List.of(embeddingId)).build(),
                FaceSearchResult.class).getBody();

        assertThat(result.getMatchedPhotos()).isNotEmpty();
        assertThat(result.getMatchedPhotos()).allSatisfy(p -> assertThat(p.getLocalPath()).isNull());
    }

    @Test
    void customerCannotAddPhotosFromAnotherEventToTheirOrder() {
        Tenant a = newTenant(2200);
        Tenant b = newTenant(2300);

        ResponseEntity<String> response = restTemplate.postForEntity(
                "/api/customer/orders/" + a.orderId() + "/items",
                AddOrderItemsRequest.builder().photoIds(List.of(b.photoId())).build(), String.class);

        assertThat(response.getStatusCode()).isEqualTo(HttpStatus.BAD_REQUEST);
    }

    @Test
    void customerCannotChangeAnOrderAfterItIsConfirmed() {
        Tenant a = newTenant(2400);

        ResponseEntity<String> confirm = restTemplate.exchange("/api/customer/orders/" + a.orderId() + "/confirm",
                HttpMethod.PUT, null, String.class);
        assertThat(confirm.getStatusCode()).isEqualTo(HttpStatus.OK);

        ResponseEntity<String> again = restTemplate.postForEntity("/api/customer/orders/" + a.orderId() + "/items",
                AddOrderItemsRequest.builder().photoIds(List.of(a.photoId())).build(), String.class);
        assertThat(again.getStatusCode()).isEqualTo(HttpStatus.CONFLICT);
    }

    // --- Input validation ---

    @Test
    void unknownOrderStatusIsRejected() {
        Tenant a = newTenant(2500);
        ResponseEntity<String> response = restTemplate.exchange(
                "/api/studio/orders/" + a.orderId() + "/status?status=HACKED", HttpMethod.PUT,
                authEntity(a.token()), String.class);
        assertThat(response.getStatusCode()).isEqualTo(HttpStatus.BAD_REQUEST);
    }

    @Test
    void photoRedirectOnlyFollowsHttpUrls() {
        Tenant a = newTenant(2600);
        RegisterPhotosRequest.PhotoEntry entry = RegisterPhotosRequest.PhotoEntry.builder()
                .filename("js.jpg").localPath("/p/js.jpg").url("javascript:alert(1)").fileHash("h-" + uniqueSuffix()).build();
        Long id = restTemplate.exchange("/api/studio/photos/register", HttpMethod.POST, authEntity(a.token(),
                RegisterPhotosRequest.builder().eventId(a.eventId()).photos(List.of(entry)).build()),
                PhotoDto[].class).getBody()[0].getId();

        ResponseEntity<String> response = restTemplate.getForEntity("/api/photos/" + id, String.class);

        assertThat(response.getStatusCode()).isEqualTo(HttpStatus.NOT_FOUND);
    }

    // --- Subscription gating ---

    @Test
    void deactivatedStudioOperatorsAreLockedOut() {
        String adminToken = loginAsSeededAdmin();
        Long studioId = createStudioAsAdmin(adminToken);
        String email = "locked-" + uniqueSuffix() + "@example.com";
        String operatorToken = restTemplate.exchange("/api/auth/register", HttpMethod.POST, authEntity(adminToken,
                RegisterRequest.builder().email(email).password("Password123!")
                        .role(AccountRole.STUDIO_OPERATOR).studioId(studioId).build()),
                AuthResponse.class).getBody().getToken();
        assertThat(get(operatorToken, "/api/studio/events?studioId=" + studioId).getStatusCode()).isEqualTo(HttpStatus.OK);

        restTemplate.exchange("/api/admin/studios/" + studioId + "/deactivate", HttpMethod.PUT,
                authEntity(adminToken), StudioDto.class);

        // Existing session is cut off...
        assertThat(get(operatorToken, "/api/studio/events?studioId=" + studioId).getStatusCode()).isEqualTo(HttpStatus.FORBIDDEN);
        // ...and a fresh login is refused.
        ResponseEntity<String> login = restTemplate.postForEntity("/api/auth/login",
                LoginRequest.builder().email(email).password("Password123!").build(), String.class);
        assertThat(login.getStatusCode()).isEqualTo(HttpStatus.FORBIDDEN);
    }
}
