import React from 'react';
import { useParams, Link } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { eventsApi, photosApi } from '../services/api';
import Spinner from '../components/Spinner';

export default function EventDetailPage() {
  const { eventId } = useParams();

  const { data: event, isLoading: eventLoading } = useQuery({
    queryKey: ['events', eventId],
    queryFn: () => eventsApi.get(eventId),
  });

  const { data: stats } = useQuery({
    queryKey: ['events', eventId, 'stats'],
    queryFn: () => photosApi.getStats(eventId),
  });

  if (eventLoading) {
    return (
      <div className="flex justify-center py-12">
        <Spinner size="lg" />
      </div>
    );
  }

  if (!event) {
    return <p className="text-gray-500">Event not found.</p>;
  }

  return (
    <div>
      <div className="flex items-center gap-3 mb-6">
        <Link to="/events" className="text-gray-400 hover:text-gray-600 text-sm">
          Events
        </Link>
        <span className="text-gray-300">/</span>
        <h1 className="page-title">{event.name}</h1>
      </div>

      {event.date && <p className="text-sm text-gray-500 mb-6">{event.date}</p>}

      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-8">
        <div className="card text-center">
          <p className="text-3xl font-bold text-primary-600">
            {stats?.totalPhotos ?? event.photoCount ?? 0}
          </p>
          <p className="text-sm text-gray-500 mt-1">Total Photos</p>
        </div>
        <div className="card text-center">
          <p className="text-3xl font-bold text-green-600">
            {stats?.processedPhotos ?? 0}
          </p>
          <p className="text-sm text-gray-500 mt-1">Processed</p>
        </div>
        <div className="card text-center">
          <p className="text-3xl font-bold text-yellow-600">
            {stats?.facesDetected ?? 0}
          </p>
          <p className="text-sm text-gray-500 mt-1">Faces Detected</p>
        </div>
      </div>

      <div className="flex gap-3">
        <Link to={`/events/${eventId}/upload`} className="btn-primary">
          Upload Photos
        </Link>
        <Link to={`/events/${eventId}/process`} className="btn-secondary">
          Process Event
        </Link>
        <Link to={`/orders/create?eventId=${eventId}`} className="btn-secondary">
          Create Order
        </Link>
      </div>
    </div>
  );
}
