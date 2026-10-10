package com.facesorter.controller;

import com.facesorter.dto.*;
import com.facesorter.security.TenantAccess;
import com.facesorter.service.FaceService;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/faces")
@RequiredArgsConstructor
public class FaceController {

    private final FaceService faceService;
    private final TenantAccess tenantAccess;

    /**
     * Submit embeddings extracted from a photo during local processing.
     * Called by the studio desktop app after running the face detection pipeline.
     */
    @PostMapping("/embeddings")
    public ResponseEntity<Void> submitEmbeddings(
            @Valid @RequestBody SubmitEmbeddingsRequest request) {
        tenantAccess.requireEvent(request.getEventId());
        faceService.submitEmbeddings(request);
        return ResponseEntity.ok().build();
    }

    /**
     * Search for matching photos in an event given a face embedding.
     */
    @PostMapping("/search")
    public ResponseEntity<FaceSearchResult> searchFaces(
            @Valid @RequestBody FaceSearchRequest request) {
        if (request.getEventId() == null) {
            throw new IllegalArgumentException("eventId is required");
        }
        tenantAccess.requireEvent(request.getEventId());
        return ResponseEntity.ok(faceService.searchFaces(request));
    }
}
