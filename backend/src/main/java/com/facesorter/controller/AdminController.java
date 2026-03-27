package com.facesorter.controller;

import com.facesorter.dto.*;
import com.facesorter.entity.*;
import com.facesorter.repository.*;
import com.facesorter.service.EventService;
import com.facesorter.service.OrderService;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.stream.Collectors;

@RestController
@RequestMapping("/api/admin")
@RequiredArgsConstructor
public class AdminController {

    private final StudioRepository studioRepository;
    private final UserRepository userRepository;
    private final SubscriptionRepository subscriptionRepository;
    private final EventService eventService;
    private final OrderService orderService;

    // --- Studios ---

    @GetMapping("/studios")
    public ResponseEntity<List<StudioDto>> getAllStudios() {
        List<StudioDto> studios = studioRepository.findAll().stream()
                .map(this::toStudioDto)
                .collect(Collectors.toList());
        return ResponseEntity.ok(studios);
    }

    @PostMapping("/studios")
    public ResponseEntity<StudioDto> createStudio(@Valid @RequestBody CreateStudioRequest request) {
        Studio studio = Studio.builder()
                .name(request.getName())
                .email(request.getEmail())
                .active(true)
                .build();
        studio = studioRepository.save(studio);
        return ResponseEntity.ok(toStudioDto(studio));
    }

    @PutMapping("/studios/{id}/activate")
    public ResponseEntity<StudioDto> activateStudio(@PathVariable Long id) {
        Studio studio = studioRepository.findById(id)
                .orElseThrow(() -> new RuntimeException("Studio not found"));
        studio.setActive(true);
        studio = studioRepository.save(studio);
        return ResponseEntity.ok(toStudioDto(studio));
    }

    @PutMapping("/studios/{id}/deactivate")
    public ResponseEntity<StudioDto> deactivateStudio(@PathVariable Long id) {
        Studio studio = studioRepository.findById(id)
                .orElseThrow(() -> new RuntimeException("Studio not found"));
        studio.setActive(false);
        studio = studioRepository.save(studio);
        return ResponseEntity.ok(toStudioDto(studio));
    }

    // --- Users ---

    @GetMapping("/users")
    public ResponseEntity<List<UserDto>> getAllUsers() {
        List<UserDto> users = userRepository.findAll().stream()
                .map(this::toUserDto)
                .collect(Collectors.toList());
        return ResponseEntity.ok(users);
    }

    @PostMapping("/users")
    public ResponseEntity<UserDto> createUser(@Valid @RequestBody CreateUserRequest request) {
        User user = User.builder()
                .idNumber(request.getIdNumber())
                .name(request.getName())
                .email(request.getEmail())
                .build();
        user = userRepository.save(user);
        return ResponseEntity.ok(toUserDto(user));
    }

    // --- Subscriptions ---

    @GetMapping("/subscriptions")
    public ResponseEntity<List<SubscriptionDto>> getAllSubscriptions() {
        List<SubscriptionDto> subs = subscriptionRepository.findAll().stream()
                .map(this::toSubscriptionDto)
                .collect(Collectors.toList());
        return ResponseEntity.ok(subs);
    }

    @PostMapping("/subscriptions")
    public ResponseEntity<SubscriptionDto> createSubscription(
            @Valid @RequestBody CreateSubscriptionRequest request) {
        Studio studio = studioRepository.findById(request.getStudioId())
                .orElseThrow(() -> new RuntimeException("Studio not found"));

        Subscription sub = Subscription.builder()
                .studio(studio)
                .startDate(request.getStartDate())
                .endDate(request.getEndDate())
                .paymentMethod(request.getPaymentMethod())
                .amountPaid(request.getAmountPaid())
                .active(true)
                .build();
        sub = subscriptionRepository.save(sub);
        return ResponseEntity.ok(toSubscriptionDto(sub));
    }

    // --- Events (cross-studio view) ---

    @GetMapping("/events")
    public ResponseEntity<List<EventDto>> getAllEvents() {
        List<EventDto> events = eventService.getEventsByStudio(null);
        // For admin, return all events
        List<Event> allEvents = new java.util.ArrayList<>();
        studioRepository.findAll().forEach(s ->
                allEvents.addAll(s.getEvents()));
        // Reuse event service for individual lookups
        return ResponseEntity.ok(events);
    }

    // --- Orders (cross-studio view) ---

    @GetMapping("/orders/{id}")
    public ResponseEntity<OrderDto> getOrder(@PathVariable Long id) {
        return ResponseEntity.ok(orderService.getOrder(id));
    }

    private StudioDto toStudioDto(Studio studio) {
        return StudioDto.builder()
                .id(studio.getId())
                .name(studio.getName())
                .email(studio.getEmail())
                .active(studio.getActive())
                .createdAt(studio.getCreatedAt())
                .build();
    }

    private UserDto toUserDto(User user) {
        return UserDto.builder()
                .id(user.getId())
                .idNumber(user.getIdNumber())
                .name(user.getName())
                .email(user.getEmail())
                .hasEmbedding(user.getFaceEmbedding() != null)
                .createdAt(user.getCreatedAt())
                .build();
    }

    private SubscriptionDto toSubscriptionDto(Subscription sub) {
        return SubscriptionDto.builder()
                .id(sub.getId())
                .studioId(sub.getStudio().getId())
                .startDate(sub.getStartDate())
                .endDate(sub.getEndDate())
                .paymentMethod(sub.getPaymentMethod())
                .amountPaid(sub.getAmountPaid())
                .active(sub.getActive())
                .build();
    }
}
