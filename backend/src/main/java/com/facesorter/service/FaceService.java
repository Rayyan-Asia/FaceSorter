package com.facesorter.service;

import com.facesorter.dto.*;
import com.facesorter.entity.*;
import com.facesorter.repository.*;
import lombok.RequiredArgsConstructor;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.*;
import java.util.stream.Collectors;

@Service
@RequiredArgsConstructor
public class FaceService {

    private final FaceEmbeddingRepository faceEmbeddingRepository;
    private final PersonPhotoLinkRepository personPhotoLinkRepository;
    private final PhotoRepository photoRepository;
    private final EventRepository eventRepository;

    @Value("${facesorter.similarity-threshold}")
    private double similarityThreshold;

    /**
     * Submit embeddings extracted from a photo by the local studio processing pipeline.
     * For each embedding, check if a matching person exists in the event; if so, link them.
     * Otherwise, create a new face embedding record.
     * Marks the photo as processed upon completion.
     */
    @Transactional
    public void submitEmbeddings(SubmitEmbeddingsRequest request) {
        Photo photo = photoRepository.findById(request.getPhotoId())
                .orElseThrow(() -> new NoSuchElementException("Photo not found: " + request.getPhotoId()));

        if (photo.getProcessed()) {
            return; // already processed, skip
        }

        Event event = eventRepository.findById(request.getEventId())
                .orElseThrow(() -> new NoSuchElementException("Event not found: " + request.getEventId()));

        for (float[] embeddingVector : request.getEmbeddings()) {
            String pgVectorLiteral = toVectorLiteral(embeddingVector);

            // Search for existing similar face in this event
            List<FaceEmbedding> matches = faceEmbeddingRepository.findSimilarInEvent(
                    event.getId(), pgVectorLiteral, similarityThreshold, 1);

            FaceEmbedding faceEmbedding;
            if (!matches.isEmpty()) {
                // Existing person found — link to this photo
                faceEmbedding = matches.get(0);
            } else {
                // New person — create embedding record
                faceEmbedding = FaceEmbedding.builder()
                        .embedding(embeddingVector)
                        .photo(photo)
                        .event(event)
                        .build();
                faceEmbedding = faceEmbeddingRepository.save(faceEmbedding);
            }

            // Create person-photo link if not already present
            if (!personPhotoLinkRepository.existsByFaceEmbeddingIdAndPhotoId(
                    faceEmbedding.getId(), photo.getId())) {
                PersonPhotoLink link = PersonPhotoLink.builder()
                        .faceEmbedding(faceEmbedding)
                        .photo(photo)
                        .build();
                personPhotoLinkRepository.save(link);
            }
        }

        photo.setProcessed(true);
        photoRepository.save(photo);
    }

    /**
     * Search for photos in an event that match a given face embedding.
     * Used by the customer portal (self-photo search) and studio operator (walk-in search).
     */
    @Transactional(readOnly = true)
    public FaceSearchResult searchFaces(FaceSearchRequest request) {
        String pgVectorLiteral = toVectorLiteral(request.getEmbedding());

        // Find all similar face embeddings in this event
        List<FaceEmbedding> similarEmbeddings = faceEmbeddingRepository.findAllSimilarInEvent(
                request.getEventId(), pgVectorLiteral, similarityThreshold);

        if (similarEmbeddings.isEmpty()) {
            return FaceSearchResult.builder()
                    .matchedPhotos(Collections.emptyList())
                    .totalMatches(0)
                    .build();
        }

        // Collect all embedding IDs, then find all photos linked to those embeddings
        List<Long> embeddingIds = similarEmbeddings.stream()
                .map(FaceEmbedding::getId)
                .toList();

        List<PersonPhotoLink> links = personPhotoLinkRepository.findByFaceEmbeddingIdIn(embeddingIds);

        // Deduplicate photos and pick the best similarity score per photo
        Map<Long, PersonPhotoLink> bestPerPhoto = new LinkedHashMap<>();
        for (PersonPhotoLink link : links) {
            Long photoId = link.getPhoto().getId();
            bestPerPhoto.merge(photoId, link, (existing, candidate) -> {
                if (candidate.getSimilarityScore() != null && existing.getSimilarityScore() != null) {
                    return candidate.getSimilarityScore() > existing.getSimilarityScore() ? candidate : existing;
                }
                return existing;
            });
        }

        List<FaceSearchResult.MatchedPhoto> matchedPhotos = bestPerPhoto.values().stream()
                .map(link -> FaceSearchResult.MatchedPhoto.builder()
                        .photoId(link.getPhoto().getId())
                        .filename(link.getPhoto().getFilename())
                        .localPath(link.getPhoto().getLocalPath())
                        .similarityScore(link.getSimilarityScore() != null ? link.getSimilarityScore() : 0.0)
                        .build())
                .collect(Collectors.toList());

        return FaceSearchResult.builder()
                .matchedPhotos(matchedPhotos)
                .totalMatches(matchedPhotos.size())
                .build();
    }

    private String toVectorLiteral(float[] embedding) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < embedding.length; i++) {
            if (i > 0) sb.append(",");
            sb.append(embedding[i]);
        }
        sb.append("]");
        return sb.toString();
    }
}
