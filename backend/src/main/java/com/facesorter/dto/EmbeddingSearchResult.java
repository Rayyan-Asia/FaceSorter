package com.facesorter.dto;

import lombok.*;
import java.util.List;

@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class EmbeddingSearchResult {
    private List<EmbeddingCandidate> candidates;

    @Getter
    @Setter
    @NoArgsConstructor
    @AllArgsConstructor
    @Builder
    public static class EmbeddingCandidate {
        private Long embeddingId;
        private Long representativePhotoId;
        private String representativeFilename;
    }
}
