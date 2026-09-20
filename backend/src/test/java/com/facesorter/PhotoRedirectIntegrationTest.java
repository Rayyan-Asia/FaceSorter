package com.facesorter;

import com.facesorter.dto.*;
import com.facesorter.entity.AccountRole;
import org.junit.jupiter.api.Test;
import org.springframework.http.HttpMethod;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.http.client.SimpleClientHttpRequestFactory;
import org.springframework.web.client.RestTemplate;

import java.io.IOException;
import java.net.HttpURLConnection;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;

class PhotoRedirectIntegrationTest extends AbstractIntegrationTest {

    /**
     * A plain RestTemplate that does NOT auto-follow redirects (so the 302 itself is
     * inspectable) and does NOT throw on 4xx (so 404s can be asserted as normal responses).
     */
    private static final RestTemplate NO_REDIRECT_CLIENT = buildNoRedirectClient();

    private static RestTemplate buildNoRedirectClient() {
        RestTemplate client = new RestTemplate(new SimpleClientHttpRequestFactory() {
            @Override
            protected void prepareConnection(HttpURLConnection connection, String httpMethod) throws IOException {
                super.prepareConnection(connection, httpMethod);
                connection.setInstanceFollowRedirects(false);
            }
        });
        client.setErrorHandler(new org.springframework.web.client.ResponseErrorHandler() {
            @Override
            public boolean hasError(org.springframework.http.client.ClientHttpResponse response) {
                return false;
            }

            @Override
            public void handleError(org.springframework.http.client.ClientHttpResponse response) {
            }
        });
        return client;
    }

    @Test
    void redirectsToPhotoUrlWithoutAuth() {
        String adminToken = loginAsSeededAdmin();
        Long studioId = restTemplate.exchange("/api/admin/studios", HttpMethod.POST,
                authEntity(adminToken, CreateStudioRequest.builder()
                        .name("Photo Studio " + uniqueSuffix()).email("photo-" + uniqueSuffix() + "@example.com").build()),
                StudioDto.class).getBody().getId();
        String operatorToken = registerAndLogin(AccountRole.STUDIO_OPERATOR, studioId);
        Long eventId = restTemplate.exchange("/api/studio/events", HttpMethod.POST,
                authEntity(operatorToken, CreateEventRequest.builder().studioId(studioId).name("Photo Event").build()),
                EventDto.class).getBody().getId();

        String photoUrl = "http://device.local/photo?path=/x.jpg";
        RegisterPhotosRequest.PhotoEntry entry = RegisterPhotosRequest.PhotoEntry.builder()
                .filename("x.jpg").localPath("/local/x.jpg").url(photoUrl).fileHash("hash-" + uniqueSuffix()).build();
        Long photoId = restTemplate.exchange("/api/studio/photos/register", HttpMethod.POST,
                authEntity(operatorToken, RegisterPhotosRequest.builder().eventId(eventId).photos(List.of(entry)).build()),
                PhotoDto[].class).getBody()[0].getId();

        ResponseEntity<Void> response = NO_REDIRECT_CLIENT.getForEntity(
                restTemplate.getRootUri() + "/api/photos/" + photoId, Void.class);

        assertThat(response.getStatusCode()).isEqualTo(HttpStatus.FOUND);
        assertThat(response.getHeaders().getLocation()).hasToString(photoUrl);
    }

    @Test
    void photoWithNoUrlReturns404() {
        String adminToken = loginAsSeededAdmin();
        Long studioId = restTemplate.exchange("/api/admin/studios", HttpMethod.POST,
                authEntity(adminToken, CreateStudioRequest.builder()
                        .name("NoUrl Studio " + uniqueSuffix()).email("nourl-" + uniqueSuffix() + "@example.com").build()),
                StudioDto.class).getBody().getId();
        String operatorToken = registerAndLogin(AccountRole.STUDIO_OPERATOR, studioId);
        Long eventId = restTemplate.exchange("/api/studio/events", HttpMethod.POST,
                authEntity(operatorToken, CreateEventRequest.builder().studioId(studioId).name("NoUrl Event").build()),
                EventDto.class).getBody().getId();

        RegisterPhotosRequest.PhotoEntry entry = RegisterPhotosRequest.PhotoEntry.builder()
                .filename("nourl.jpg").localPath("/local/nourl.jpg").url(null).fileHash("hash-" + uniqueSuffix()).build();
        Long photoId = restTemplate.exchange("/api/studio/photos/register", HttpMethod.POST,
                authEntity(operatorToken, RegisterPhotosRequest.builder().eventId(eventId).photos(List.of(entry)).build()),
                PhotoDto[].class).getBody()[0].getId();

        ResponseEntity<Void> response = NO_REDIRECT_CLIENT.getForEntity(
                restTemplate.getRootUri() + "/api/photos/" + photoId, Void.class);

        assertThat(response.getStatusCode()).isEqualTo(HttpStatus.NOT_FOUND);
    }

    @Test
    void unknownPhotoIdReturns404() {
        ResponseEntity<String> response = restTemplate.getForEntity("/api/photos/999999", String.class);
        assertThat(response.getStatusCode()).isEqualTo(HttpStatus.NOT_FOUND);
    }
}
