import axios from 'axios';
import { useAuthStore } from '../store/authStore';

const API_BASE_URL = 'http://localhost:8080/api';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

api.interceptors.request.use((config) => {
  const token = useAuthStore.getState().token;
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      useAuthStore.getState().logout();
    }
    return Promise.reject(error);
  }
);

// Events
export const eventsApi = {
  list: () => api.get('/events').then((r) => r.data),
  get: (id) => api.get(`/events/${id}`).then((r) => r.data),
  create: (data) => api.post('/events', data).then((r) => r.data),
  delete: (id) => api.delete(`/events/${id}`),
};

// Photos
export const photosApi = {
  listByEvent: (eventId, processed) => {
    const params = processed !== undefined ? { processed } : {};
    return api.get(`/events/${eventId}/photos`, { params }).then((r) => r.data);
  },
  register: (eventId, photos) =>
    api.post(`/events/${eventId}/photos`, { photos }).then((r) => r.data),
  getStats: (eventId) =>
    api.get(`/events/${eventId}/photos/stats`).then((r) => r.data),
};

// Orders
export const ordersApi = {
  list: (params) => api.get('/orders', { params }).then((r) => r.data),
  get: (id) => api.get(`/orders/${id}`).then((r) => r.data),
  create: (data) => api.post('/orders', data).then((r) => r.data),
  updateStatus: (id, status) =>
    api.put(`/orders/${id}/status`, { status }).then((r) => r.data),
  addItems: (orderId, photoIds) =>
    api.post(`/orders/${orderId}/items`, { photoIds }).then((r) => r.data),
};

// Face matching
export const matchingApi = {
  search: (eventId, embedding) =>
    api.post(`/events/${eventId}/search`, { embedding }).then((r) => r.data),
};

// Auth
export const authApi = {
  googleLogin: (credential) =>
    api.post('/auth/google', { credential }).then((r) => r.data),
  me: () => api.get('/auth/me').then((r) => r.data),
};

export { API_BASE_URL };
export default api;
