package com.facesorter.controller;

import com.facesorter.dto.*;
import com.facesorter.entity.Event;
import com.facesorter.security.TenantAccess;
import com.facesorter.service.EventService;
import com.facesorter.service.OrderService;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;

/**
 * Studio operator API. Every handler verifies the caller may access the studio that owns the
 * target event/order/photo (see {@link TenantAccess}) before delegating to the service.
 */
@RestController
@RequestMapping("/api/studio")
@RequiredArgsConstructor
public class StudioController {

    private final EventService eventService;
    private final OrderService orderService;
    private final TenantAccess tenantAccess;

    // --- Events ---

    @PostMapping("/events")
    public ResponseEntity<EventDto> createEvent(@Valid @RequestBody CreateEventRequest request) {
        tenantAccess.requireStudio(request.getStudioId());
        return ResponseEntity.ok(eventService.createEvent(request));
    }

    @GetMapping("/events")
    public ResponseEntity<List<EventDto>> getEvents(@RequestParam Long studioId) {
        tenantAccess.requireStudio(studioId);
        return ResponseEntity.ok(eventService.getEventsByStudio(studioId));
    }

    @GetMapping("/events/{eventId}")
    public ResponseEntity<EventDto> getEvent(@PathVariable Long eventId) {
        tenantAccess.requireEvent(eventId);
        return ResponseEntity.ok(eventService.getEvent(eventId));
    }

    // --- Photos ---

    @PostMapping("/photos/register")
    public ResponseEntity<List<PhotoDto>> registerPhotos(
            @Valid @RequestBody RegisterPhotosRequest request) {
        tenantAccess.requireEvent(request.getEventId());
        return ResponseEntity.ok(eventService.registerPhotos(request));
    }

    @GetMapping("/events/{eventId}/photos")
    public ResponseEntity<List<PhotoDto>> getPhotos(@PathVariable Long eventId) {
        tenantAccess.requireEvent(eventId);
        return ResponseEntity.ok(eventService.getPhotos(eventId));
    }

    @GetMapping("/events/{eventId}/photos/unprocessed")
    public ResponseEntity<List<PhotoDto>> getUnprocessedPhotos(@PathVariable Long eventId) {
        tenantAccess.requireEvent(eventId);
        return ResponseEntity.ok(eventService.getUnprocessedPhotos(eventId));
    }

    // --- Orders ---

    @PostMapping("/orders")
    public ResponseEntity<OrderDto> createOrder(@Valid @RequestBody CreateOrderRequest request) {
        Event event = tenantAccess.requireEvent(request.getEventId());
        if (request.getStudioId() != null) {
            tenantAccess.requireStudio(request.getStudioId());
            if (!request.getStudioId().equals(event.getStudio().getId())) {
                throw new IllegalArgumentException("Order studio must match the event's studio");
            }
        }
        return ResponseEntity.ok(orderService.createOrder(request));
    }

    @GetMapping("/orders")
    public ResponseEntity<List<OrderDto>> getOrders(@RequestParam Long studioId) {
        tenantAccess.requireStudio(studioId);
        return ResponseEntity.ok(orderService.getOrdersByStudio(studioId));
    }

    @GetMapping("/orders/{orderId}")
    public ResponseEntity<OrderDto> getOrder(@PathVariable Long orderId) {
        tenantAccess.requireOrder(orderId);
        return ResponseEntity.ok(orderService.getOrder(orderId));
    }

    @PutMapping("/orders/{orderId}/status")
    public ResponseEntity<OrderDto> updateOrderStatus(
            @PathVariable Long orderId, @RequestParam String status) {
        tenantAccess.requireOrder(orderId);
        return ResponseEntity.ok(orderService.updateOrderStatus(orderId, status));
    }

    @PostMapping("/orders/{orderId}/items")
    public ResponseEntity<OrderDto> addOrderItems(
            @PathVariable Long orderId, @Valid @RequestBody AddOrderItemsRequest request) {
        tenantAccess.requireOrder(orderId);
        return ResponseEntity.ok(orderService.addOrderItems(orderId, request));
    }
}
