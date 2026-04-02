import React from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { ordersApi } from '../services/api';
import { useAuthStore } from '../store/authStore';
import Spinner from '../components/Spinner';
import EmptyState from '../components/EmptyState';

const STATUS_FLOW = {
  PENDING: { next: 'CONFIRMED', label: 'Confirm Order', btnClass: 'btn-primary' },
  CONFIRMED: { next: 'FULFILLED', label: 'Mark as Printed', btnClass: 'btn-primary' },
};

export default function OrderFulfillmentPage() {
  const queryClient = useQueryClient();
  const user = useAuthStore((s) => s.user);

  const { data: fetchedOrders, isLoading } = useQuery({
    queryKey: ['orders', user?.studioId],
    queryFn: () => ordersApi.list(user?.studioId),
  });

  const statusMutation = useMutation({
    mutationFn: ({ id, status }) => ordersApi.updateStatus(id, status),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['orders'] });
    },
  });

  const allOrders = (fetchedOrders || [])
    .filter((o) => o.status === 'PENDING' || o.status === 'CONFIRMED')
    .map((o) => ({ ...o, _section: o.status }));

  return (
    <div>
      <h1 className="page-title mb-6">Order Fulfillment</h1>

      {isLoading && (
        <div className="flex justify-center py-12">
          <Spinner size="lg" />
        </div>
      )}

      {!isLoading && allOrders.length === 0 && (
        <EmptyState
          title="No orders to fulfill"
          description="All caught up. Confirmed orders will appear here for printing."
        />
      )}

      {allOrders.length > 0 && (
        <div className="space-y-3">
          {allOrders.map((order) => {
            const flow = STATUS_FLOW[order.status];
            return (
              <div key={order.id} className="card flex items-center justify-between">
                <div>
                  <div className="flex items-center gap-3">
                    <span className="font-bold text-lg">#{order.id}</span>
                    <span
                      className={
                        order.status === 'CONFIRMED' ? 'badge-blue' : 'badge-yellow'
                      }
                    >
                      {order.status}
                    </span>
                  </div>
                  <p className="text-sm text-gray-500 mt-1">
                    Event: {order.eventName || order.eventId}
                    {order.customerName && ` | ${order.customerName}`}
                    {order.itemCount != null && ` | ${order.itemCount} items`}
                  </p>
                </div>

                {flow && (
                  <button
                    onClick={() =>
                      statusMutation.mutate({ id: order.id, status: flow.next })
                    }
                    disabled={statusMutation.isPending}
                    className={flow.btnClass}
                  >
                    {statusMutation.isPending &&
                    statusMutation.variables?.id === order.id
                      ? 'Updating...'
                      : flow.label}
                  </button>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
