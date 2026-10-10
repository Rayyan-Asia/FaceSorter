package com.facesorter.security;

import com.facesorter.entity.Account;
import com.facesorter.entity.AccountRole;
import com.facesorter.entity.Event;
import com.facesorter.entity.Order;
import com.facesorter.entity.Photo;
import com.facesorter.repository.AccountRepository;
import com.facesorter.repository.EventRepository;
import com.facesorter.repository.OrderRepository;
import com.facesorter.repository.PhotoRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.security.access.AccessDeniedException;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.stereotype.Component;
import org.springframework.transaction.annotation.Transactional;

import java.util.NoSuchElementException;

/**
 * Multi-tenant authorization: a studio operator may only touch their own studio's data.
 * Admins are unrestricted. Every authenticated studio/face endpoint must go through here,
 * otherwise any operator could read or modify another studio's events, photos and orders
 * just by guessing sequential ids.
 */
@Component
@RequiredArgsConstructor
public class TenantAccess {

    private final AccountRepository accountRepository;
    private final EventRepository eventRepository;
    private final OrderRepository orderRepository;
    private final PhotoRepository photoRepository;

    private Account currentAccount() {
        Authentication auth = SecurityContextHolder.getContext().getAuthentication();
        if (auth == null || !(auth.getPrincipal() instanceof AccountDetails details)) {
            throw new AccessDeniedException("Authentication required");
        }
        // Re-load inside the caller's transaction: the principal's account is detached, so its
        // lazy studio can't be read, and this also picks up deletion/changes since login.
        return accountRepository.findByEmail(details.getUsername())
                .orElseThrow(() -> new AccessDeniedException("Account no longer exists"));
    }

    /** Allows admins, or operators whose own (active) studio matches {@code studioId}. */
    @Transactional(readOnly = true)
    public void requireStudio(Long studioId) {
        Account account = currentAccount();
        if (account.getRole() == AccountRole.ADMIN) {
            return;
        }
        if (account.getStudio() == null || studioId == null
                || !account.getStudio().getId().equals(studioId)) {
            throw new AccessDeniedException("Not allowed to access this studio");
        }
        if (!Boolean.TRUE.equals(account.getStudio().getActive())) {
            throw new AccessDeniedException("Studio is deactivated");
        }
    }

    @Transactional(readOnly = true)
    public Event requireEvent(Long eventId) {
        Event event = eventRepository.findById(eventId)
                .orElseThrow(() -> new NoSuchElementException("Event not found: " + eventId));
        requireStudio(event.getStudio().getId());
        return event;
    }

    @Transactional(readOnly = true)
    public Order requireOrder(Long orderId) {
        Order order = orderRepository.findById(orderId)
                .orElseThrow(() -> new NoSuchElementException("Order not found: " + orderId));
        requireStudio(order.getStudio().getId());
        return order;
    }

    @Transactional(readOnly = true)
    public Photo requirePhoto(Long photoId) {
        Photo photo = photoRepository.findById(photoId)
                .orElseThrow(() -> new NoSuchElementException("Photo not found: " + photoId));
        requireStudio(photo.getEvent().getStudio().getId());
        return photo;
    }
}
