package com.facesorter.dto;

import lombok.*;
import java.time.LocalDateTime;

@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class UserDto {
    private Long id;
    private String idNumber;
    private String name;
    private String email;
    private boolean hasEmbedding;
    private LocalDateTime createdAt;
}
