package com.facesorter.dto;

import lombok.*;
import java.time.LocalDateTime;

@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class PhotoDto {
    private Long id;
    private Long eventId;
    private String filename;
    private String localPath;
    private String url;
    private Boolean processed;
    private LocalDateTime createdAt;
}
