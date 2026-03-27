package com.facesorter.dto;

import lombok.*;
import java.util.List;

@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class FaceSearchResult {
    private List<MatchedPhoto> matchedPhotos;
    private int totalMatches;

    @Getter
    @Setter
    @NoArgsConstructor
    @AllArgsConstructor
    @Builder
    public static class MatchedPhoto {
        private Long photoId;
        private String filename;
        private String localPath;
        private double similarityScore;
    }
}
