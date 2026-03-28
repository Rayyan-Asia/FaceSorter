package com.facesorter.controller;

import com.facesorter.entity.Photo;
import com.facesorter.repository.PhotoRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.NoSuchElementException;

@RestController
@RequestMapping("/api/photos")
@RequiredArgsConstructor
public class PhotoController {

    private final PhotoRepository photoRepository;

    /**
     * Redirect to the photo's origin URL, which is served by the studio device
     * that registered the photo. Each studio device runs its own local file server.
     */
    @GetMapping("/{id}")
    public ResponseEntity<Void> getPhoto(@PathVariable Long id) {
        Photo photo = photoRepository.findById(id)
                .orElseThrow(() -> new NoSuchElementException("Photo not found: " + id));

        if (photo.getUrl() == null || photo.getUrl().isBlank()) {
            return ResponseEntity.notFound().build();
        }

        return ResponseEntity.status(HttpStatus.FOUND)
                .header(HttpHeaders.LOCATION, photo.getUrl())
                .build();
    }
}
