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
 * Exercises the permitAll customer-facing order/search/select flow with no auth token,
 * exactly as a customer-portal client would call it.
 */
class CustomerFlowIntegrationTest extends AbstractIntegrationTest {

    private record TestOrder(Long eventId, Long orderId, Long photoId, float[] embedding) {}

    private TestOrder setupOrderWithOnePhoto() {
        String adminToken = loginAsSeededAdmin();
        Long studioId = restTemplate.exchange("/api/admin/studios", HttpMethod.POST,
                authEntity(adminToken, CreateStudioRequest.builder()
                        .name("Cust Studio " + uniqueSuffix()).email("cust-" + uniqueSuffix() + "@example.com").build()),
                StudioDto.class).getBody().getId();

        String operatorToken = registerAndLogin(AccountRole.STUDIO_OPERATOR, studioId);
        Long eventId = restTemplate.exchange("/api/studio/events", HttpMethod.POST,
                authEntity(operatorToken, CreateEventRequest.builder().studioId(studioId).name("Cust Event").build()),
                EventDto.class).getBody().getId();

        RegisterPhotosRequest.PhotoEntry entry = RegisterPhotosRequest.PhotoEntry.builder()
                .filename("cust.jpg").localPath("/local/cust.jpg").url("http://device/cust.jpg")
                .fileHash("hash-" + uniqueSuffix()).build();
        Long photoId = restTemplate.exchange("/api/studio/photos/register", HttpMethod.POST,
                authEntity(operatorToken, RegisterPhotosRequest.builder().eventId(eventId).photos(List.of(entry)).build()),
                PhotoDto[].class).getBody()[0].getId();

        float[] embedding = identityVector(700);
        restTemplate.postForEntity("/api/faces/embeddings", SubmitEmbeddingsRequest.builder()
                .photoId(photoId).eventId(eventId).embeddings(List.of(embedding)).build(), Void.class);

        Long orderId = restTemplate.exchange("/api/studio/orders", HttpMethod.POST,
                authEntity(operatorToken, CreateOrderRequest.builder().eventId(eventId).build()), OrderDto.class)
                .getBody().getId();

        return new TestOrder(eventId, orderId, photoId, embedding);
    }

    @Test
    void customerCanAccessOrderWithoutAnyToken() {
        TestOrder ctx = setupOrderWithOnePhoto();
        ResponseEntity<OrderDto> response = restTemplate.getForEntity("/api/customer/orders/" + ctx.orderId(), OrderDto.class);

        assertThat(response.getStatusCode()).isEqualTo(HttpStatus.OK);
        assertThat(response.getBody().getEventId()).isEqualTo(ctx.eventId());
    }

    @Test
    void accessingUnknownOrderReturns404() {
        ResponseEntity<String> response = restTemplate.getForEntity("/api/customer/orders/999999", String.class);
        assertThat(response.getStatusCode()).isEqualTo(HttpStatus.NOT_FOUND);
    }

    @Test
    void fullSelfServeFlow_searchSelectConfirm() {
        TestOrder ctx = setupOrderWithOnePhoto();

        // Step 1: customer submits only their embedding — no eventId needed, it's derived
        // from the order server-side. (Regression test: this used to 400 because
        // FaceSearchRequest.eventId was @NotNull.)
        FaceSearchRequest step1 = FaceSearchRequest.builder().embedding(ctx.embedding()).build();
        ResponseEntity<EmbeddingSearchResult> step1Response = restTemplate.postForEntity(
                "/api/customer/orders/" + ctx.orderId() + "/search", step1, EmbeddingSearchResult.class);

        assertThat(step1Response.getStatusCode()).isEqualTo(HttpStatus.OK);
        List<Long> candidateIds = step1Response.getBody().getCandidates().stream()
                .map(EmbeddingSearchResult.EmbeddingCandidate::getEmbeddingId).toList();
        assertThat(candidateIds).hasSize(1);

        // Step 2: fetch all photos linked to the confirmed candidate embeddings
        PhotosByEmbeddingsRequest step2 = PhotosByEmbeddingsRequest.builder().embeddingIds(candidateIds).build();
        ResponseEntity<FaceSearchResult> step2Response = restTemplate.postForEntity(
                "/api/customer/orders/" + ctx.orderId() + "/photos", step2, FaceSearchResult.class);

        assertThat(step2Response.getStatusCode()).isEqualTo(HttpStatus.OK);
        assertThat(step2Response.getBody().getMatchedPhotos())
                .extracting(FaceSearchResult.MatchedPhoto::getPhotoId)
                .containsExactly(ctx.photoId());

        // Customer selects the photo
        AddOrderItemsRequest select = AddOrderItemsRequest.builder().photoIds(List.of(ctx.photoId())).build();
        ResponseEntity<OrderDto> selectResponse = restTemplate.postForEntity(
                "/api/customer/orders/" + ctx.orderId() + "/items", select, OrderDto.class);
        assertThat(selectResponse.getStatusCode()).isEqualTo(HttpStatus.OK);
        assertThat(selectResponse.getBody().getOrderItems()).hasSize(1);
        assertThat(selectResponse.getBody().getOrderItems().get(0).getPhotoIds()).containsExactly(ctx.photoId());

        // Customer confirms
        ResponseEntity<OrderDto> confirmResponse = restTemplate.exchange(
                "/api/customer/orders/" + ctx.orderId() + "/confirm", HttpMethod.PUT, null, OrderDto.class);
        assertThat(confirmResponse.getStatusCode()).isEqualTo(HttpStatus.OK);
        assertThat(confirmResponse.getBody().getStatus()).isEqualTo("CONFIRMED");
    }
}
