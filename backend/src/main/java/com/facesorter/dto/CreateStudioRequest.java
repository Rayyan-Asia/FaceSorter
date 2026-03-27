package com.facesorter.dto;

import jakarta.validation.constraints.Email;
import jakarta.validation.constraints.NotBlank;
import lombok.*;

@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class CreateStudioRequest {
    @NotBlank
    private String name;

    @NotBlank
    @Email
    private String email;
}
