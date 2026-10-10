package com.facesorter.security;

import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;
import org.springframework.web.filter.OncePerRequestFilter;

import java.io.IOException;
import java.util.Iterator;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Per-client-IP fixed-window rate limiting for the endpoints reachable without a token:
 * login (credential guessing) and /api/customer/** (order-id enumeration + face search).
 * Limits are generous for real use (a studio counter, a customer taking a selfie) and are
 * configurable; set a very high value to disable in tests.
 */
@Component
public class RateLimitFilter extends OncePerRequestFilter {

    private static final long WINDOW_MS = 60_000L;

    private final int loginPerMinute;
    private final int customerPerMinute;
    private final Map<String, Window> windows = new ConcurrentHashMap<>();

    public RateLimitFilter(
            @Value("${facesorter.rate-limit.login-per-minute:20}") int loginPerMinute,
            @Value("${facesorter.rate-limit.customer-per-minute:120}") int customerPerMinute) {
        this.loginPerMinute = loginPerMinute;
        this.customerPerMinute = customerPerMinute;
    }

    @Override
    protected void doFilterInternal(HttpServletRequest request, HttpServletResponse response,
                                    FilterChain chain) throws ServletException, IOException {
        String path = request.getRequestURI();
        int limit;
        String bucket;
        if (path.equals("/api/auth/login")) {
            limit = loginPerMinute;
            bucket = "login";
        } else if (path.startsWith("/api/customer/")) {
            limit = customerPerMinute;
            bucket = "customer";
        } else {
            chain.doFilter(request, response);
            return;
        }

        // remoteAddr, not X-Forwarded-For: that header is client-controlled and would let an
        // attacker dodge the limit by rotating it.
        String key = bucket + ":" + request.getRemoteAddr();
        long now = System.currentTimeMillis();
        evictExpired(now);
        Window w = windows.compute(key, (k, cur) ->
                (cur == null || now - cur.start >= WINDOW_MS) ? new Window(now) : cur);
        if (w.count.incrementAndGet() > limit) {
            response.setStatus(429);
            response.setHeader("Retry-After", String.valueOf(WINDOW_MS / 1000));
            return;
        }
        chain.doFilter(request, response);
    }

    private void evictExpired(long now) {
        if (windows.size() < 10_000) return;
        for (Iterator<Map.Entry<String, Window>> it = windows.entrySet().iterator(); it.hasNext(); ) {
            if (now - it.next().getValue().start >= WINDOW_MS) it.remove();
        }
    }

    private static final class Window {
        final long start;
        final java.util.concurrent.atomic.AtomicInteger count = new java.util.concurrent.atomic.AtomicInteger();
        Window(long start) { this.start = start; }
    }
}
