package com.facesorter.dto;

import jakarta.validation.constraints.NotNull;
import lombok.*;

@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class FaceSearchRequest {
    @NotNull
    private Long eventId;

    @NotNull
    private float[] embedding;
}
