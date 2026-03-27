package com.facesorter.dto;

import jakarta.validation.constraints.NotNull;
import lombok.*;

@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class CreateOrderRequest {
    @NotNull
    private Long eventId;

    @NotNull
    private Long studioId;

    private Long userId;
}
