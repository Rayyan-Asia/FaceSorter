import React from 'react';
import { HashRouter, Routes, Route, Navigate } from 'react-router-dom';
import { useAuthStore } from './store/authStore';
import Layout from './components/Layout';
import LoginPage from './pages/LoginPage';
import EventsPage from './pages/EventsPage';
import EventDetailPage from './pages/EventDetailPage';
import PhotoUploadPage from './pages/PhotoUploadPage';
import ProcessEventPage from './pages/ProcessEventPage';
import OrdersPage from './pages/OrdersPage';
import CreateOrderPage from './pages/CreateOrderPage';
import WalkInRetrievalPage from './pages/WalkInRetrievalPage';
import OrderFulfillmentPage from './pages/OrderFulfillmentPage';

function ProtectedRoute({ children }) {
  const isAuthenticated = useAuthStore((s) => s.isAuthenticated);
  if (!isAuthenticated) return <Navigate to="/login" replace />;
  return children;
}

export default function App() {
  return (
    <HashRouter>
      <Routes>
        <Route path="/login" element={<LoginPage />} />
        <Route
          path="/"
          element={
            <ProtectedRoute>
              <Layout />
            </ProtectedRoute>
          }
        >
          <Route index element={<Navigate to="/events" replace />} />
          <Route path="events" element={<EventsPage />} />
          <Route path="events/:eventId" element={<EventDetailPage />} />
          <Route path="events/:eventId/upload" element={<PhotoUploadPage />} />
          <Route path="events/:eventId/process" element={<ProcessEventPage />} />
          <Route path="orders" element={<OrdersPage />} />
          <Route path="orders/create" element={<CreateOrderPage />} />
          <Route path="orders/:orderId/walk-in" element={<WalkInRetrievalPage />} />
          <Route path="orders/fulfillment" element={<OrderFulfillmentPage />} />
        </Route>
      </Routes>
    </HashRouter>
  );
}
