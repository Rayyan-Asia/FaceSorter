import React, { useState, useRef, useEffect } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { useQuery, useMutation } from '@tanstack/react-query';
import { eventsApi, ordersApi } from '../services/api';
import { useAuthStore } from '../store/authStore';
import Spinner from '../components/Spinner';

export default function CreateOrderPage() {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const preselectedEventId = searchParams.get('eventId') || '';
  const user = useAuthStore((s) => s.user);

  const [selectedEventId, setSelectedEventId] = useState(preselectedEventId);
  const [createdOrder, setCreatedOrder] = useState(null);
  const [search, setSearch] = useState('');
  const [open, setOpen] = useState(false);
  const comboRef = useRef(null);

  useEffect(() => {
    const handler = (e) => {
      if (comboRef.current && !comboRef.current.contains(e.target)) setOpen(false);
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, []);

  const { data: events, isLoading: eventsLoading } = useQuery({
    queryKey: ['events', user?.studioId],
    queryFn: () => eventsApi.list(user?.studioId),
  });

  const createMutation = useMutation({
    mutationFn: ordersApi.create,
    onSuccess: (data) => {
      setCreatedOrder(data);
    },
  });

  const handleCreate = (e) => {
    e.preventDefault();
    if (!selectedEventId) return;
    createMutation.mutate({
      eventId: Number(selectedEventId),
    });
  };

  if (createdOrder) {
    return (
      <div>
        <h1 className="page-title mb-6">Order Created</h1>
        <div className="card text-center py-8">
          <p className="text-sm text-gray-500 mb-2">Order ID</p>
          <p className="text-5xl font-bold text-primary-600 mb-4">
            #{createdOrder.id}
          </p>
          <p className="text-sm text-gray-500 mb-6">
            Order created. Proceed to retrieve the customer's photos in the next step.
          </p>
          <div className="flex gap-3 justify-center">
            <button
              onClick={() => setCreatedOrder(null)}
              className="btn-secondary"
            >
              Create Another
            </button>
            <button
              onClick={() => navigate(`/orders/${createdOrder.id}/walk-in`)}
              className="btn-primary"
            >
              Start Photo Retrieval
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div>
      <h1 className="page-title mb-6">Create Order</h1>

      <div className="card max-w-lg">
        <p className="text-sm text-gray-500 mb-6">
          Create an empty order for a walk-in customer. The customer pays first, then
          receives the order ID to retrieve their photos.
        </p>

        <form onSubmit={handleCreate} className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Event</label>
            {eventsLoading ? (
              <Spinner size="sm" />
            ) : (
              <div ref={comboRef} className="relative">
                <input
                  type="text"
                  className="input-field"
                  placeholder={selectedEventId
                    ? events?.find((ev) => String(ev.id) === String(selectedEventId))?.name ?? 'Select an event'
                    : 'Search events...'}
                  value={open ? search : (events?.find((ev) => String(ev.id) === String(selectedEventId))?.name ?? '')}
                  onFocus={() => { setOpen(true); setSearch(''); }}
                  onChange={(e) => { setSearch(e.target.value); setOpen(true); }}
                  required={!selectedEventId}
                />
                {open && (
                  <ul className="absolute z-10 mt-1 w-full bg-white border border-gray-200 rounded-lg shadow-lg max-h-56 overflow-y-auto">
                    {(events ?? [])
                      .filter((ev) => ev.name.toLowerCase().includes(search.toLowerCase()))
                      .map((ev) => (
                        <li
                          key={ev.id}
                          onMouseDown={() => {
                            setSelectedEventId(String(ev.id));
                            setOpen(false);
                            setSearch('');
                          }}
                          className={`px-3 py-2 cursor-pointer text-sm hover:bg-primary-50 ${
                            String(ev.id) === String(selectedEventId) ? 'bg-primary-50 font-medium' : ''
                          }`}
                        >
                          {ev.name}{ev.eventDate ? ` (${ev.eventDate})` : ''}
                        </li>
                      ))}
                    {(events ?? []).filter((ev) => ev.name.toLowerCase().includes(search.toLowerCase())).length === 0 && (
                      <li className="px-3 py-2 text-sm text-gray-400">No events found</li>
                    )}
                  </ul>
                )}
              </div>
            )}
          </div>

          <button
            type="submit"
            disabled={createMutation.isPending || !selectedEventId}
            className="btn-primary w-full"
          >
            {createMutation.isPending ? 'Creating...' : 'Create Order'}
          </button>

          {createMutation.isError && (
            <p className="text-sm text-red-600">
              Failed to create order: {createMutation.error?.message}
            </p>
          )}
        </form>
      </div>
    </div>
  );
}
