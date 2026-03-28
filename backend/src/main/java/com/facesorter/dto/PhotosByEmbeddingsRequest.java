package com.facesorter.dto;

import jakarta.validation.constraints.NotEmpty;
import lombok.*;
import java.util.List;

@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class PhotosByEmbeddingsRequest {
    @NotEmpty
    private List<Long> embeddingIds;
}
