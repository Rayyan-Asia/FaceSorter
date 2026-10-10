package com.facesorter.security;

import jakarta.servlet.FilterChain;
import org.junit.jupiter.api.Test;
import org.springframework.mock.web.MockHttpServletRequest;
import org.springframework.mock.web.MockHttpServletResponse;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.mock;

class RateLimitFilterTest {

    private int hit(RateLimitFilter filter, String path, String ip) throws Exception {
        MockHttpServletRequest req = new MockHttpServletRequest("POST", path);
        req.setRequestURI(path);
        req.setRemoteAddr(ip);
        MockHttpServletResponse res = new MockHttpServletResponse();
        filter.doFilter(req, res, mock(FilterChain.class));
        return res.getStatus();
    }

    @Test
    void blocksLoginAfterLimitPerClientIp() throws Exception {
        RateLimitFilter filter = new RateLimitFilter(3, 100);

        assertThat(hit(filter, "/api/auth/login", "10.0.0.1")).isEqualTo(200);
        assertThat(hit(filter, "/api/auth/login", "10.0.0.1")).isEqualTo(200);
        assertThat(hit(filter, "/api/auth/login", "10.0.0.1")).isEqualTo(200);
        assertThat(hit(filter, "/api/auth/login", "10.0.0.1")).isEqualTo(429);
        // A different client is unaffected.
        assertThat(hit(filter, "/api/auth/login", "10.0.0.2")).isEqualTo(200);
    }

    @Test
    void limitsCustomerEndpointsSeparatelyFromLogin() throws Exception {
        RateLimitFilter filter = new RateLimitFilter(1, 2);

        assertThat(hit(filter, "/api/customer/orders/1", "10.0.0.1")).isEqualTo(200);
        assertThat(hit(filter, "/api/customer/orders/2", "10.0.0.1")).isEqualTo(200);
        assertThat(hit(filter, "/api/customer/orders/3", "10.0.0.1")).isEqualTo(429);
        // Login bucket is independent.
        assertThat(hit(filter, "/api/auth/login", "10.0.0.1")).isEqualTo(200);
    }

    @Test
    void ignoresOtherPaths() throws Exception {
        RateLimitFilter filter = new RateLimitFilter(1, 1);
        for (int i = 0; i < 10; i++) {
            assertThat(hit(filter, "/api/studio/events", "10.0.0.1")).isEqualTo(200);
        }
    }
}
