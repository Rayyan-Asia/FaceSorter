package com.facesorter.dto;

import jakarta.validation.constraints.NotNull;
import lombok.*;

@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class FaceSearchRequest {
    // Not @NotNull: the customer-facing search endpoint derives this from the
    // order server-side, so the client isn't required to supply it.
    private Long eventId;

    @NotNull
    private float[] embedding;

    /** Optional per-request override for the similarity threshold. Null = use server default. */
    private Double threshold;
}
