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

    /** Public callers must not learn the studio machine's filesystem layout. */
    private static FaceSearchResult withoutLocalPaths(FaceSearchResult result) {
        result.getMatchedPhotos().forEach(p -> p.setLocalPath(null));
        return result;
    }

    /**
     * Step 1 of face-based photo retrieval.
     * Customer submits their face embedding; returns the top 10 most similar face embeddings
     * found in the event as candidates for the customer to confirm are them.
     */
    @PostMapping("/orders/{orderId}/search")
    public ResponseEntity<EmbeddingSearchResult> searchByFace(
            @PathVariable Long orderId,
            @Valid @RequestBody FaceSearchRequest request) {
        OrderDto order = orderService.getOrder(orderId);
        request.setEventId(order.getEventId());
        return ResponseEntity.ok(faceService.searchEmbeddings(request));
    }

    /**
     * Step 2 of face-based photo retrieval.
     * Customer confirms which embedding IDs are them; returns all photos linked to those embeddings.
     */
    @PostMapping("/orders/{orderId}/photos")
    public ResponseEntity<FaceSearchResult> getPhotosByEmbeddings(
            @PathVariable Long orderId,
            @Valid @RequestBody PhotosByEmbeddingsRequest request) {
        OrderDto order = orderService.getOrder(orderId);
        return ResponseEntity.ok(withoutLocalPaths(
                faceService.getPhotosByEmbeddingIds(order.getEventId(), request.getEmbeddingIds())));
    }

    /**
     * Customer selects photos and confirms the order.
     */
    @PostMapping("/orders/{orderId}/items")
    public ResponseEntity<OrderDto> selectPhotos(
            @PathVariable Long orderId,
            @Valid @RequestBody AddOrderItemsRequest request) {
        orderService.requireOpenForCustomer(orderId);
        return ResponseEntity.ok(orderService.addOrderItems(orderId, request));
    }

    /**
     * Customer finalizes order after selecting photos.
     */
    @PutMapping("/orders/{orderId}/confirm")
    public ResponseEntity<OrderDto> confirmOrder(@PathVariable Long orderId) {
        orderService.requireOpenForCustomer(orderId);
        return ResponseEntity.ok(orderService.updateOrderStatus(orderId, "CONFIRMED"));
    }
}
