package com.facesorter.dto;

import lombok.*;
import java.time.LocalDate;
import java.time.LocalDateTime;

@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class EventDto {
    private Long id;
    private Long studioId;
    private String studioName;
    private String name;
    private String description;
    private LocalDate eventDate;
    private long totalPhotos;
    private long processedPhotos;
    private LocalDateTime createdAt;
}
