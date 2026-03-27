package com.facesorter.dto;

import jakarta.validation.constraints.NotNull;
import lombok.*;
import java.util.List;

@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class AddOrderItemsRequest {
    @NotNull
    private List<Long> photoIds;

    private String notes;
}
