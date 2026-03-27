package com.facesorter.dto;

import jakarta.validation.constraints.NotNull;
import lombok.*;
import java.util.List;

@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class SubmitEmbeddingsRequest {
    @NotNull
    private Long photoId;

    @NotNull
    private Long eventId;

    @NotNull
    private List<float[]> embeddings;
}
