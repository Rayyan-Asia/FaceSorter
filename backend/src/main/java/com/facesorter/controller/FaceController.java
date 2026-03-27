package com.facesorter.controller;

import com.facesorter.dto.*;
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

    /**
     * Submit embeddings extracted from a photo during local processing.
     * Called by the studio desktop app after running the face detection pipeline.
     */
    @PostMapping("/embeddings")
    public ResponseEntity<Void> submitEmbeddings(
            @Valid @RequestBody SubmitEmbeddingsRequest request) {
        faceService.submitEmbeddings(request);
        return ResponseEntity.ok().build();
    }

    /**
     * Search for matching photos in an event given a face embedding.
     */
    @PostMapping("/search")
    public ResponseEntity<FaceSearchResult> searchFaces(
            @Valid @RequestBody FaceSearchRequest request) {
        return ResponseEntity.ok(faceService.searchFaces(request));
    }
}
