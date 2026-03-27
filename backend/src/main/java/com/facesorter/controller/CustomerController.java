package com.facesorter.controller;

import com.facesorter.dto.*;
import com.facesorter.service.FaceService;
import com.facesorter.service.OrderService;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/customer")
@RequiredArgsConstructor
public class CustomerController {

    private final OrderService orderService;
    private final FaceService faceService;

    /**
     * Customer enters their order ID to access the linked event.
     */
    @GetMapping("/orders/{orderId}")
    public ResponseEntity<OrderDto> accessOrder(@PathVariable Long orderId) {
        return ResponseEntity.ok(orderService.getOrder(orderId));
    }

    /**
     * Customer takes a self-photo; the frontend extracts the embedding and sends it here.
     * Returns all matching photos from the event linked to their order.
     */
    @PostMapping("/orders/{orderId}/search")
    public ResponseEntity<FaceSearchResult> searchByFace(
            @PathVariable Long orderId,
            @Valid @RequestBody FaceSearchRequest request) {
        // Ensure the search is scoped to the event linked to this order
        OrderDto order = orderService.getOrder(orderId);
        request.setEventId(order.getEventId());

        return ResponseEntity.ok(faceService.searchFaces(request));
    }

    /**
     * Customer selects photos and confirms the order.
     */
    @PostMapping("/orders/{orderId}/items")
    public ResponseEntity<OrderDto> selectPhotos(
            @PathVariable Long orderId,
            @Valid @RequestBody AddOrderItemsRequest request) {
        return ResponseEntity.ok(orderService.addOrderItems(orderId, request));
    }

    /**
     * Customer finalizes order after selecting photos.
     */
    @PutMapping("/orders/{orderId}/confirm")
    public ResponseEntity<OrderDto> confirmOrder(@PathVariable Long orderId) {
        return ResponseEntity.ok(orderService.updateOrderStatus(orderId, "CONFIRMED"));
    }
}
