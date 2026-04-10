package com.facesorter.service;

import com.facesorter.dto.*;
import com.facesorter.entity.*;
import com.facesorter.repository.*;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;
import java.util.NoSuchElementException;
import java.util.Set;
import java.util.stream.Collectors;

@Service
@RequiredArgsConstructor
public class EventService {

    private final EventRepository eventRepository;
    private final StudioRepository studioRepository;
    private final PhotoRepository photoRepository;

    @Transactional
    public EventDto createEvent(CreateEventRequest request) {
        Studio studio = studioRepository.findById(request.getStudioId())
                .orElseThrow(() -> new NoSuchElementException("Studio not found: " + request.getStudioId()));

        Event event = Event.builder()
                .studio(studio)
                .name(request.getName())
                .description(request.getDescription())
                .eventDate(request.getEventDate())
                .build();

        event = eventRepository.save(event);
        return toDto(event);
    }

    @Transactional(readOnly = true)
    public EventDto getEvent(Long eventId) {
        Event event = eventRepository.findById(eventId)
                .orElseThrow(() -> new NoSuchElementException("Event not found: " + eventId));
        return toDto(event);
    }

    @Transactional(readOnly = true)
    public List<EventDto> getEventsByStudio(Long studioId) {
        return eventRepository.findByStudioId(studioId).stream()
                .map(this::toDto)
                .collect(Collectors.toList());
    }

    @Transactional(readOnly = true)
    public List<EventDto> getAllEvents() {
        return eventRepository.findAll().stream()
                .map(this::toDto)
                .collect(Collectors.toList());
    }

    @Transactional
    public List<PhotoDto> registerPhotos(RegisterPhotosRequest request) {
        Event event = eventRepository.findById(request.getEventId())
                .orElseThrow(() -> new NoSuchElementException("Event not found: " + request.getEventId()));

        Set<String> existingHashes = photoRepository.findFileHashByEventIdAndFileHashNotNull(event.getId());

        List<Photo> photos = request.getPhotos().stream()
                .filter(entry -> entry.getFileHash() == null || !existingHashes.contains(entry.getFileHash()))
                .map(entry -> Photo.builder()
                        .event(event)
                        .filename(entry.getFilename())
                        .localPath(entry.getLocalPath())
                        .url(entry.getUrl())
                        .fileHash(entry.getFileHash())
                        .processed(false)
                        .build())
                .collect(Collectors.toList());

        photos = photoRepository.saveAll(photos);

        return photos.stream()
                .map(this::toPhotoDto)
                .collect(Collectors.toList());
    }

    @Transactional(readOnly = true)
    public List<PhotoDto> getUnprocessedPhotos(Long eventId) {
        return photoRepository.findByEventIdAndProcessedFalse(eventId).stream()
                .map(this::toPhotoDto)
                .collect(Collectors.toList());
    }

    @Transactional(readOnly = true)
    public List<PhotoDto> getPhotos(Long eventId) {
        return photoRepository.findByEventId(eventId).stream()
                .map(this::toPhotoDto)
                .collect(Collectors.toList());
    }

    private EventDto toDto(Event event) {
        return EventDto.builder()
                .id(event.getId())
                .studioId(event.getStudio().getId())
                .studioName(event.getStudio().getName())
                .name(event.getName())
                .description(event.getDescription())
                .eventDate(event.getEventDate())
                .totalPhotos(photoRepository.countByEventId(event.getId()))
                .processedPhotos(photoRepository.countByEventIdAndProcessedTrue(event.getId()))
                .createdAt(event.getCreatedAt())
                .build();
    }

    private PhotoDto toPhotoDto(Photo photo) {
        return PhotoDto.builder()
                .id(photo.getId())
                .eventId(photo.getEvent().getId())
                .filename(photo.getFilename())
                .localPath(photo.getLocalPath())
                .url(photo.getUrl())
                .processed(photo.getProcessed())
                .createdAt(photo.getCreatedAt())
                .build();
    }
}
