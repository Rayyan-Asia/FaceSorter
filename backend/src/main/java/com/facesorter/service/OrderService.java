package com.facesorter.service;

import com.facesorter.dto.*;
import com.facesorter.entity.*;
import com.facesorter.repository.*;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;
import java.util.NoSuchElementException;
import java.util.stream.Collectors;

@Service
@RequiredArgsConstructor
public class OrderService {

    private final OrderRepository orderRepository;
    private final OrderItemRepository orderItemRepository;
    private final OrderItemPhotoRepository orderItemPhotoRepository;
    private final PhotoRepository photoRepository;
    private final EventRepository eventRepository;
    private final StudioRepository studioRepository;
    private final UserRepository userRepository;

    /**
     * Create an empty order linked to an event.
     * Called by studio operator at point of payment.
     */
    @Transactional
    public OrderDto createOrder(CreateOrderRequest request) {
        Event event = eventRepository.findById(request.getEventId())
                .orElseThrow(() -> new NoSuchElementException("Event not found: " + request.getEventId()));

        // Use the provided studioId, or fall back to the event's own studio
        Studio studio = (request.getStudioId() != null)
                ? studioRepository.findById(request.getStudioId())
                        .orElseThrow(() -> new NoSuchElementException("Studio not found: " + request.getStudioId()))
                : event.getStudio();

        Order order = Order.builder()
                .event(event)
                .studio(studio)
                .status("CREATED")
                .build();

        if (request.getUserId() != null) {
            User user = userRepository.findById(request.getUserId())
                    .orElseThrow(() -> new NoSuchElementException("User not found: " + request.getUserId()));
            order.setUser(user);
        }

        order = orderRepository.save(order);
        return toDto(order);
    }

    @Transactional(readOnly = true)
    public OrderDto getOrder(Long orderId) {
        Order order = orderRepository.findById(orderId)
                .orElseThrow(() -> new NoSuchElementException("Order not found: " + orderId));
        return toDto(order);
    }

    @Transactional(readOnly = true)
    public List<OrderDto> getAllOrders() {
        return orderRepository.findAll().stream()
                .map(this::toDto)
                .collect(Collectors.toList());
    }

    @Transactional(readOnly = true)
    public List<OrderDto> getOrdersByStudio(Long studioId) {
        return orderRepository.findByStudioId(studioId).stream()
                .map(this::toDto)
                .collect(Collectors.toList());
    }

    @Transactional(readOnly = true)
    public List<OrderDto> getOrdersByEvent(Long eventId) {
        return orderRepository.findByEventId(eventId).stream()
                .map(this::toDto)
                .collect(Collectors.toList());
    }

    /**
     * Add selected photos to an order as order items.
     * Called by customer after selecting their photos, or by studio operator.
     */
    @Transactional
    public OrderDto addOrderItems(Long orderId, AddOrderItemsRequest request) {
        Order order = orderRepository.findById(orderId)
                .orElseThrow(() -> new NoSuchElementException("Order not found: " + orderId));

        OrderItem orderItem = OrderItem.builder()
                .order(order)
                .notes(request.getNotes())
                .build();
        orderItem = orderItemRepository.save(orderItem);

        for (Long photoId : request.getPhotoIds()) {
            Photo photo = photoRepository.findById(photoId)
                    .orElseThrow(() -> new NoSuchElementException("Photo not found: " + photoId));

            OrderItemPhoto oip = OrderItemPhoto.builder()
                    .orderItem(orderItem)
                    .photo(photo)
                    .build();
            orderItemPhotoRepository.save(oip);
        }

        return toDto(orderRepository.findById(orderId).orElseThrow());
    }

    @Transactional
    public OrderDto updateOrderStatus(Long orderId, String status) {
        Order order = orderRepository.findById(orderId)
                .orElseThrow(() -> new NoSuchElementException("Order not found: " + orderId));

        order.setStatus(status);
        order = orderRepository.save(order);
        return toDto(order);
    }

    private OrderDto toDto(Order order) {
        List<OrderItemDto> items = orderItemRepository.findByOrderId(order.getId()).stream()
                .map(item -> {
                    List<Long> photoIds = orderItemPhotoRepository.findByOrderItemId(item.getId()).stream()
                            .map(oip -> oip.getPhoto().getId())
                            .collect(Collectors.toList());

                    return OrderItemDto.builder()
                            .id(item.getId())
                            .orderId(order.getId())
                            .notes(item.getNotes())
                            .photoIds(photoIds)
                            .build();
                })
                .collect(Collectors.toList());

        return OrderDto.builder()
                .id(order.getId())
                .userId(order.getUser() != null ? order.getUser().getId() : null)
                .eventId(order.getEvent().getId())
                .eventName(order.getEvent().getName())
                .studioId(order.getStudio().getId())
                .status(order.getStatus())
                .orderItems(items)
                .createdAt(order.getCreatedAt())
                .updatedAt(order.getUpdatedAt())
                .build();
    }
}
