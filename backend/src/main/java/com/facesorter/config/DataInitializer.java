package com.facesorter.config;

import com.facesorter.entity.Account;
import com.facesorter.entity.AccountRole;
import com.facesorter.repository.AccountRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.boot.CommandLineRunner;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Component;

@Slf4j
@Component
@RequiredArgsConstructor
public class DataInitializer implements CommandLineRunner {

    private final AccountRepository accountRepository;
    private final PasswordEncoder passwordEncoder;

    /** Initial admin password; override with ADMIN_INITIAL_PASSWORD for any non-local deployment. */
    @org.springframework.beans.factory.annotation.Value("${facesorter.admin.initial-password:Admin1234!}")
    private String initialAdminPassword;

    @Override
    public void run(String... args) {
        if (accountRepository.findByEmail("admin@facesorter.com").isEmpty()) {
            Account admin = Account.builder()
                    .email("admin@facesorter.com")
                    .passwordHash(passwordEncoder.encode(initialAdminPassword))
                    .role(AccountRole.ADMIN)
                    .build();
            accountRepository.save(admin);
            log.info("Seeded default admin account: admin@facesorter.com (change its password after first login)");
        }
    }
}
