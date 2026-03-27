package com.facesorter.controller;

import com.facesorter.dto.*;
import com.facesorter.service.EventService;
import com.facesorter.service.OrderService;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/studio")
@RequiredArgsConstructor
public class StudioController {

    private final EventService eventService;
    private final OrderService orderService;

    // --- Events ---

    @PostMapping("/events")
    public ResponseEntity<EventDto> createEvent(@Valid @RequestBody CreateEventRequest request) {
        return ResponseEntity.ok(eventService.createEvent(request));
    }

    @GetMapping("/events")
    public ResponseEntity<List<EventDto>> getEvents(@RequestParam Long studioId) {
        return ResponseEntity.ok(eventService.getEventsByStudio(studioId));
    }

    @GetMapping("/events/{eventId}")
    public ResponseEntity<EventDto> getEvent(@PathVariable Long eventId) {
        return ResponseEntity.ok(eventService.getEvent(eventId));
    }

    // --- Photos ---

    @PostMapping("/photos/register")
    public ResponseEntity<List<PhotoDto>> registerPhotos(
            @Valid @RequestBody RegisterPhotosRequest request) {
        return ResponseEntity.ok(eventService.registerPhotos(request));
    }

    @GetMapping("/events/{eventId}/photos")
    public ResponseEntity<List<PhotoDto>> getPhotos(@PathVariable Long eventId) {
        return ResponseEntity.ok(eventService.getPhotos(eventId));
    }

    @GetMapping("/events/{eventId}/photos/unprocessed")
    public ResponseEntity<List<PhotoDto>> getUnprocessedPhotos(@PathVariable Long eventId) {
        return ResponseEntity.ok(eventService.getUnprocessedPhotos(eventId));
    }

    // --- Orders ---

    @PostMapping("/orders")
    public ResponseEntity<OrderDto> createOrder(@Valid @RequestBody CreateOrderRequest request) {
        return ResponseEntity.ok(orderService.createOrder(request));
    }

    @GetMapping("/orders")
    public ResponseEntity<List<OrderDto>> getOrders(@RequestParam Long studioId) {
        return ResponseEntity.ok(orderService.getOrdersByStudio(studioId));
    }

    @GetMapping("/orders/{orderId}")
    public ResponseEntity<OrderDto> getOrder(@PathVariable Long orderId) {
        return ResponseEntity.ok(orderService.getOrder(orderId));
    }

    @PutMapping("/orders/{orderId}/status")
    public ResponseEntity<OrderDto> updateOrderStatus(
            @PathVariable Long orderId, @RequestParam String status) {
        return ResponseEntity.ok(orderService.updateOrderStatus(orderId, status));
    }

    @PostMapping("/orders/{orderId}/items")
    public ResponseEntity<OrderDto> addOrderItems(
            @PathVariable Long orderId, @Valid @RequestBody AddOrderItemsRequest request) {
        return ResponseEntity.ok(orderService.addOrderItems(orderId, request));
    }
}
