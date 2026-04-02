package com.facesorter.controller;

import com.facesorter.dto.*;
import com.facesorter.entity.Studio;
import com.facesorter.entity.Subscription;
import com.facesorter.entity.User;
import com.facesorter.repository.*;
import com.facesorter.service.EventService;
import com.facesorter.service.OrderService;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

@RestController
@RequestMapping("/api/admin")
@RequiredArgsConstructor
public class AdminController {

    private final StudioRepository studioRepository;
    private final UserRepository userRepository;
    private final SubscriptionRepository subscriptionRepository;
    private final EventRepository eventRepository;
    private final OrderRepository orderRepository;
    private final EventService eventService;
    private final OrderService orderService;

    // --- Dashboard ---

    @GetMapping("/dashboard/stats")
    public ResponseEntity<Map<String, Long>> getDashboardStats() {
        Map<String, Long> stats = Map.of(
                "totalStudios", studioRepository.count(),
                "totalUsers", userRepository.count(),
                "totalEvents", eventRepository.count(),
                "activeSubscriptions", subscriptionRepository.countByActiveTrue(),
                "totalOrders", orderRepository.count()
        );
        return ResponseEntity.ok(stats);
    }

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

    @Transactional(readOnly = true)
    @GetMapping("/subscriptions")
    public ResponseEntity<List<SubscriptionDto>> getAllSubscriptions() {
        List<SubscriptionDto> subs = subscriptionRepository.findAll().stream()
                .map(this::toSubscriptionDto)
                .collect(Collectors.toList());
        return ResponseEntity.ok(subs);
    }

    @Transactional
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
        return ResponseEntity.ok(eventService.getAllEvents());
    }

    // --- Orders (cross-studio view) ---

    @GetMapping("/orders")
    public ResponseEntity<List<OrderDto>> getAllOrders() {
        return ResponseEntity.ok(orderService.getAllOrders());
    }

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
        String status;
        if (!Boolean.TRUE.equals(sub.getActive())) {
            status = "expired";
        } else if (sub.getEndDate() != null && sub.getEndDate().isBefore(java.time.LocalDate.now())) {
            status = "expired";
        } else {
            status = "active";
        }
        return SubscriptionDto.builder()
                .id(sub.getId())
                .studioId(sub.getStudio().getId())
                .studioName(sub.getStudio().getName())
                .plan("annual")
                .status(status)
                .startDate(sub.getStartDate())
                .endDate(sub.getEndDate())
                .paymentMethod(sub.getPaymentMethod())
                .amountPaid(sub.getAmountPaid())
                .active(sub.getActive())
                .build();
    }
}
