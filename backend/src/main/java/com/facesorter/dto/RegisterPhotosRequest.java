package com.facesorter.dto;

import jakarta.validation.constraints.NotNull;
import lombok.*;
import java.util.List;

@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class RegisterPhotosRequest {
    @NotNull
    private Long eventId;

    @NotNull
    private List<PhotoEntry> photos;

    @Getter
    @Setter
    @NoArgsConstructor
    @AllArgsConstructor
    @Builder
    public static class PhotoEntry {
        private String filename;
        private String localPath;
        private String url;
        private String fileHash;
    }
}
