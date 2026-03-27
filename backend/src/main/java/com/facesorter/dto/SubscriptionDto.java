package com.facesorter.dto;

import lombok.*;
import java.math.BigDecimal;
import java.time.LocalDate;

@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class SubscriptionDto {
    private Long id;
    private Long studioId;
    private LocalDate startDate;
    private LocalDate endDate;
    private String paymentMethod;
    private BigDecimal amountPaid;
    private Boolean active;
}
