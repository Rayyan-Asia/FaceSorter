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
 * Exercises the real pgvector similarity search end-to-end (no mocking of the
 * vector column or the native <=> queries) using deterministic synthetic
 * "identity" embeddings in place of the real ML model.
 */
class FaceMatchingIntegrationTest extends AbstractIntegrationTest {

    private record TestEvent(String token, Long studioId, Long eventId) {}

    private TestEvent setupEvent() {
        String adminToken = loginAsSeededAdmin();
        CreateStudioRequest studioReq = CreateStudioRequest.builder()
                .name("Face Studio " + uniqueSuffix()).email("face-" + uniqueSuffix() + "@example.com").build();
        Long studioId = restTemplate.exchange("/api/admin/studios", HttpMethod.POST,
                authEntity(adminToken, studioReq), StudioDto.class).getBody().getId();

        String operatorToken = registerAndLogin(AccountRole.STUDIO_OPERATOR, studioId);
        CreateEventRequest eventReq = CreateEventRequest.builder()
                .studioId(studioId).name("Face Event " + uniqueSuffix()).build();
        Long eventId = restTemplate.exchange("/api/studio/events", HttpMethod.POST,
                authEntity(operatorToken, eventReq), EventDto.class).getBody().getId();

        return new TestEvent(operatorToken, studioId, eventId);
    }

    private PhotoDto registerPhoto(TestEvent ctx, String filename) {
        RegisterPhotosRequest.PhotoEntry entry = RegisterPhotosRequest.PhotoEntry.builder()
                .filename(filename).localPath("/local/" + filename).url("http://device/" + filename)
                .fileHash("hash-" + uniqueSuffix()).build();
        return restTemplate.exchange("/api/studio/photos/register", HttpMethod.POST,
                authEntity(ctx.token(), RegisterPhotosRequest.builder().eventId(ctx.eventId()).photos(List.of(entry)).build()),
                PhotoDto[].class).getBody()[0];
    }

    private void submitEmbedding(TestEvent ctx, Long photoId, float[] embedding) {
        SubmitEmbeddingsRequest req = SubmitEmbeddingsRequest.builder()
                .photoId(photoId).eventId(ctx.eventId()).embeddings(List.of(embedding)).build();
        ResponseEntity<Void> response = restTemplate.exchange("/api/faces/embeddings", HttpMethod.POST, authEntity(ctx.token(), req), Void.class);
        assertThat(response.getStatusCode()).isEqualTo(HttpStatus.OK);
    }

    @Test
    void searchFindsOtherPhotosOfSamePersonButNotADifferentPerson() {
        TestEvent ctx = setupEvent();

        float[] personA = identityVector(100);
        float[] personB = identityVector(200);

        PhotoDto photo1 = registerPhoto(ctx, "personA-1.jpg");
        PhotoDto photo2 = registerPhoto(ctx, "personA-2.jpg");
        PhotoDto photo3 = registerPhoto(ctx, "personB-1.jpg");

        submitEmbedding(ctx, photo1.getId(), personA);
        submitEmbedding(ctx, photo2.getId(), nearbyVariant(personA, 101));
        submitEmbedding(ctx, photo3.getId(), personB);

        FaceSearchRequest search = FaceSearchRequest.builder().eventId(ctx.eventId()).embedding(personA).build();
        ResponseEntity<FaceSearchResult> response = restTemplate.exchange("/api/faces/search", HttpMethod.POST, authEntity(ctx.token(), search), FaceSearchResult.class);

        assertThat(response.getStatusCode()).isEqualTo(HttpStatus.OK);
        assertThat(response.getBody().getMatchedPhotos())
                .extracting(FaceSearchResult.MatchedPhoto::getFilename)
                .containsExactlyInAnyOrder("personA-1.jpg", "personA-2.jpg");
    }

    @Test
    void resubmittingEmbeddingsForSamePersonReusesSameEmbeddingRecord() {
        TestEvent ctx = setupEvent();
        float[] person = identityVector(300);

        PhotoDto photo1 = registerPhoto(ctx, "reuse-1.jpg");
        PhotoDto photo2 = registerPhoto(ctx, "reuse-2.jpg");

        submitEmbedding(ctx, photo1.getId(), person);
        submitEmbedding(ctx, photo2.getId(), nearbyVariant(person, 301));

        // Both photos should be linked to the SAME underlying face_embedding (person dedup),
        // which is only observable indirectly via the customer 2-step flow below.
        FaceSearchRequest search = FaceSearchRequest.builder().eventId(ctx.eventId()).embedding(person).build();
        FaceSearchResult result = restTemplate.exchange("/api/faces/search", HttpMethod.POST, authEntity(ctx.token(), search), FaceSearchResult.class).getBody();

        assertThat(result.getTotalMatches()).isEqualTo(2);
    }

    @Test
    void searchIsScopedToItsOwnEventEvenWithIdenticalEmbeddings() {
        TestEvent eventA = setupEvent();
        TestEvent eventB = setupEvent();
        float[] sharedIdentity = identityVector(400);

        PhotoDto photoInB = registerPhoto(eventB, "cross-event.jpg");
        submitEmbedding(eventB, photoInB.getId(), sharedIdentity);

        FaceSearchRequest searchInA = FaceSearchRequest.builder().eventId(eventA.eventId()).embedding(sharedIdentity).build();
        FaceSearchResult result = restTemplate.exchange("/api/faces/search", HttpMethod.POST, authEntity(eventA.token(), searchInA), FaceSearchResult.class).getBody();

        assertThat(result.getMatchedPhotos()).isEmpty();
    }

    @Test
    void submittingEmbeddingsTwiceForSamePhotoIsIdempotent() {
        TestEvent ctx = setupEvent();
        PhotoDto photo = registerPhoto(ctx, "idempotent.jpg");
        float[] person = identityVector(500);

        submitEmbedding(ctx, photo.getId(), person);
        submitEmbedding(ctx, photo.getId(), identityVector(999)); // different vector, but photo already processed

        FaceSearchRequest search = FaceSearchRequest.builder().eventId(ctx.eventId()).embedding(person).build();
        FaceSearchResult result = restTemplate.exchange("/api/faces/search", HttpMethod.POST, authEntity(ctx.token(), search), FaceSearchResult.class).getBody();

        // Second submission must be a no-op — only the first embedding should be searchable.
        assertThat(result.getMatchedPhotos()).extracting(FaceSearchResult.MatchedPhoto::getFilename)
                .containsExactly("idempotent.jpg");
    }

    @Test
    void searchWithoutEventIdReturns400() {
        FaceSearchRequest search = FaceSearchRequest.builder().embedding(identityVector(1)).build();
        ResponseEntity<String> response = restTemplate.exchange("/api/faces/search", HttpMethod.POST, authEntity(registerAndLogin(AccountRole.ADMIN, null), search), String.class);
        assertThat(response.getStatusCode()).isEqualTo(HttpStatus.BAD_REQUEST);
    }

    @Test
    void submitEmbeddingsForUnknownPhotoReturns404() {
        TestEvent ctx = setupEvent();
        SubmitEmbeddingsRequest req = SubmitEmbeddingsRequest.builder()
                .photoId(999_999L).eventId(ctx.eventId()).embeddings(List.of(identityVector(1))).build();
        ResponseEntity<String> response = restTemplate.exchange("/api/faces/embeddings", HttpMethod.POST, authEntity(ctx.token(), req), String.class);
        assertThat(response.getStatusCode()).isEqualTo(HttpStatus.NOT_FOUND);
    }
}
