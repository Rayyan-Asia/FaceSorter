import React, { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { Link } from 'react-router-dom';
import { eventsApi } from '../services/api';
import { useAuthStore } from '../store/authStore';
import Spinner from '../components/Spinner';
import EmptyState from '../components/EmptyState';

export default function EventsPage() {
  const queryClient = useQueryClient();
  const user = useAuthStore((s) => s.user);
  const [showCreate, setShowCreate] = useState(false);
  const [newEventName, setNewEventName] = useState('');
  const [newEventDate, setNewEventDate] = useState('');

  const { data: events, isLoading, error } = useQuery({
    queryKey: ['events', user?.studioId],
    queryFn: () => eventsApi.list(user?.studioId),
  });

  const createMutation = useMutation({
    mutationFn: eventsApi.create,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['events'] });
      setShowCreate(false);
      setNewEventName('');
      setNewEventDate('');
    },
  });

  const handleCreate = (e) => {
    e.preventDefault();
    if (!newEventName.trim()) return;
    createMutation.mutate({
      studioId: user?.studioId,
      name: newEventName.trim(),
      eventDate: newEventDate || undefined,
    });
  };

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <h1 className="page-title">Events</h1>
        <button onClick={() => setShowCreate(!showCreate)} className="btn-primary">
          {showCreate ? 'Cancel' : 'New Event'}
        </button>
      </div>

      {showCreate && (
        <form onSubmit={handleCreate} className="card mb-6">
          <h2 className="text-lg font-semibold mb-4">Create Event</h2>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Event Name</label>
              <input
                type="text"
                value={newEventName}
                onChange={(e) => setNewEventName(e.target.value)}
                className="input-field"
                placeholder="e.g. Ahmed's Wedding"
                required
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Date</label>
              <input
                type="date"
                value={newEventDate}
                onChange={(e) => setNewEventDate(e.target.value)}
                className="input-field"
              />
            </div>
          </div>
          <div className="mt-4">
            <button
              type="submit"
              disabled={createMutation.isPending}
              className="btn-primary"
            >
              {createMutation.isPending ? 'Creating...' : 'Create Event'}
            </button>
          </div>
          {createMutation.isError && (
            <p className="mt-2 text-sm text-red-600">
              Failed to create event: {createMutation.error?.message}
            </p>
          )}
        </form>
      )}

      {isLoading && (
        <div className="flex justify-center py-12">
          <Spinner size="lg" />
        </div>
      )}

      {error && (
        <div className="card text-center py-8">
          <p className="text-red-600">Failed to load events. Is the backend running?</p>
          <p className="text-sm text-gray-500 mt-1">{error.message}</p>
        </div>
      )}

      {events && events.length === 0 && (
        <EmptyState
          title="No events yet"
          description="Create your first event to get started."
        />
      )}

      {events && events.length > 0 && (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {events.map((event) => (
            <Link
              key={event.id}
              to={`/events/${event.id}`}
              className="card hover:shadow-md transition-shadow"
            >
              <h3 className="font-semibold text-gray-900">{event.name}</h3>
              {event.date && (
                <p className="text-sm text-gray-500 mt-1">{event.date}</p>
              )}
              <div className="flex gap-2 mt-3">
                <span className="badge-blue">
                  {event.totalPhotos ?? 0} photos
                </span>
                {event.processedPhotos != null && (
                  <span className="badge-green">
                    {event.processedPhotos} processed
                  </span>
                )}
              </div>
            </Link>
          ))}
        </div>
      )}
    </div>
  );
}
