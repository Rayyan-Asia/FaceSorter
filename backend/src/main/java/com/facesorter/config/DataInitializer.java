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

    @Override
    public void run(String... args) {
        if (accountRepository.findByEmail("admin@facesorter.com").isEmpty()) {
            Account admin = Account.builder()
                    .email("admin@facesorter.com")
                    .passwordHash(passwordEncoder.encode("Admin1234!"))
                    .role(AccountRole.ADMIN)
                    .build();
            accountRepository.save(admin);
            log.info("Seeded default admin account: admin@facesorter.com / Admin1234!");
        }
    }
}
