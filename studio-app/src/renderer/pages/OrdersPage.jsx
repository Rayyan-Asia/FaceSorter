import React, { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Link } from 'react-router-dom';
import { ordersApi } from '../services/api';
import Spinner from '../components/Spinner';
import EmptyState from '../components/EmptyState';

const STATUS_BADGE = {
  PENDING: 'badge-yellow',
  CONFIRMED: 'badge-blue',
  FULFILLED: 'badge-green',
  CANCELLED: 'badge-gray',
};

export default function OrdersPage() {
  const [statusFilter, setStatusFilter] = useState('');

  const { data: orders, isLoading, error } = useQuery({
    queryKey: ['orders', statusFilter],
    queryFn: () => ordersApi.list(statusFilter ? { status: statusFilter } : {}),
  });

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <h1 className="page-title">Orders</h1>
        <Link to="/orders/create" className="btn-primary">
          New Order
        </Link>
      </div>

      <div className="flex gap-2 mb-4">
        {['', 'PENDING', 'CONFIRMED', 'FULFILLED'].map((status) => (
          <button
            key={status}
            onClick={() => setStatusFilter(status)}
            className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
              statusFilter === status
                ? 'bg-primary-600 text-white'
                : 'bg-white text-gray-600 border border-gray-300 hover:bg-gray-50'
            }`}
          >
            {status || 'All'}
          </button>
        ))}
      </div>

      {isLoading && (
        <div className="flex justify-center py-12">
          <Spinner size="lg" />
        </div>
      )}

      {error && (
        <div className="card text-center py-8">
          <p className="text-red-600">Failed to load orders.</p>
        </div>
      )}

      {orders && orders.length === 0 && (
        <EmptyState
          title="No orders found"
          description={statusFilter ? `No orders with status "${statusFilter}".` : 'Create your first order to get started.'}
        />
      )}

      {orders && orders.length > 0 && (
        <div className="space-y-3">
          {orders.map((order) => (
            <div key={order.id} className="card flex items-center justify-between">
              <div>
                <div className="flex items-center gap-3">
                  <span className="font-bold text-lg">#{order.id}</span>
                  <span className={STATUS_BADGE[order.status] || 'badge-gray'}>
                    {order.status}
                  </span>
                </div>
                <p className="text-sm text-gray-500 mt-1">
                  Event: {order.eventName || order.eventId}
                  {order.customerName && ` | ${order.customerName}`}
                </p>
                {order.createdAt && (
                  <p className="text-xs text-gray-400 mt-0.5">
                    {new Date(order.createdAt).toLocaleString()}
                  </p>
                )}
              </div>
              <div className="flex gap-2">
                {order.status === 'PENDING' && (
                  <Link
                    to={`/orders/${order.id}/walk-in`}
                    className="btn-secondary text-sm"
                  >
                    Walk-in Retrieval
                  </Link>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
