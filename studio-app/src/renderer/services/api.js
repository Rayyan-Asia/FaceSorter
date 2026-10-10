import axios from 'axios';
import { useAuthStore, isTokenExpired } from '../store/authStore';

const API_BASE_URL = 'http://localhost:8080/api';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

api.interceptors.request.use((config) => {
  const { token, logout } = useAuthStore.getState();
  if (token) {
    if (isTokenExpired(token)) {
      // Expired while the app was open (e.g. machine slept past the timer): clear the
      // session now instead of sending a doomed request.
      logout();
      return Promise.reject(new axios.CanceledError('Session expired'));
    }
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

// The local photo server only serves directories it has been told about. Photos registered
// before that restriction existed are made viewable again by telling it which directories
// the backend already lists for this studio.
async function trustPhotoPaths(photos) {
  const paths = (photos || []).map((p) => p?.localPath).filter(Boolean);
  if (paths.length && window.electronAPI?.trustPhotoPaths) {
    try { await window.electronAPI.trustPhotoPaths({ paths }); } catch { /* non-fatal */ }
  }
}

// Events
export const eventsApi = {
  list: (studioId) => studioId
    ? api.get('/studio/events', { params: { studioId } }).then((r) => r.data)
    : api.get('/admin/events').then((r) => r.data),
  get: (id) => api.get(`/studio/events/${id}`).then((r) => r.data),
  create: (data) => api.post('/studio/events', data).then((r) => r.data),
  delete: (id) => api.delete(`/studio/events/${id}`),
};

// Photos
export const photosApi = {
  listByEvent: (eventId) =>
    api.get(`/studio/events/${eventId}/photos`).then(async (r) => {
      await trustPhotoPaths(r.data);
      return r.data;
    }),
  listUnprocessed: (eventId) =>
    api.get(`/studio/events/${eventId}/photos/unprocessed`).then(async (r) => {
      await trustPhotoPaths(r.data);
      return r.data;
    }),
  register: (eventId, photos) =>
    api.post('/studio/photos/register', { eventId, photos }).then((r) => r.data),
};

// Orders
export const ordersApi = {
  list: (studioId) => studioId
    ? api.get('/studio/orders', { params: { studioId } }).then((r) => r.data)
    : api.get('/admin/orders').then((r) => r.data),
  get: (id) => api.get(`/studio/orders/${id}`).then((r) => r.data),
  create: (data) => api.post('/studio/orders', data).then((r) => r.data),
  updateStatus: (id, status) =>
    api.put(`/studio/orders/${id}/status`, null, { params: { status } }).then((r) => r.data),
  addItems: (orderId, photoIds) =>
    api.post(`/studio/orders/${orderId}/items`, { photoIds }).then((r) => r.data),
};

// Face matching
export const matchingApi = {
  search: (eventId, embedding, threshold) =>
    api.post('/faces/search', { eventId, embedding, threshold }).then(async (r) => {
      await trustPhotoPaths(r.data?.matchedPhotos);
      return r.data;
    }),
};

// Auth
export const authApi = {
  login: (email, password) =>
    api.post('/auth/login', { email, password }).then((r) => r.data),
  me: () => api.get('/auth/me').then((r) => r.data),
};

export { API_BASE_URL };
export default api;
