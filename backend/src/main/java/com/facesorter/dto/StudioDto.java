package com.facesorter.dto;

import lombok.*;
import java.time.LocalDateTime;

@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class StudioDto {
    private Long id;
    private String name;
    private String email;
    private Boolean active;
    private LocalDateTime createdAt;
}
