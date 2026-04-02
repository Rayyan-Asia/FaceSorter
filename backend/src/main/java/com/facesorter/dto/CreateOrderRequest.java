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

    private Long studioId; // optional — derived from the event's studio if null

    private Long userId;
}
