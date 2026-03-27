package com.facesorter.dto;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import lombok.*;
import java.time.LocalDate;

@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class CreateEventRequest {
    @NotNull
    private Long studioId;

    @NotBlank
    private String name;

    private String description;
    private LocalDate eventDate;
}
