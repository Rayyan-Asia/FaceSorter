import axios from "axios";

const client = axios.create({
  baseURL: "/api/admin",
  headers: { "Content-Type": "application/json" },
});

// --- Types ---

export interface Studio {
  id: number;
  name: string;
  email: string;
  subscriptionStatus: "active" | "expired" | "none";
  createdAt: string;
}

export interface User {
  id: number;
  name: string;
  email: string;
  idNumber: string;
  createdAt: string;
}

export interface Event {
  id: number;
  name: string;
  studioId: number;
  studioName: string;
  photoCount: number;
  processedCount: number;
  createdAt: string;
}

export interface Subscription {
  id: number;
  studioId: number;
  studioName: string;
  plan: string;
  status: "active" | "expired" | "pending";
  startDate: string;
  endDate: string;
  paymentMethod: "lahza" | "cash";
}

export interface DashboardStats {
  totalStudios: number;
  totalUsers: number;
  totalEvents: number;
  activeSubscriptions: number;
  totalOrders: number;
}

export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  page: number;
  pageSize: number;
}

// --- Dashboard ---

export async function getDashboardStats(): Promise<DashboardStats> {
  const { data } = await client.get<DashboardStats>("/dashboard/stats");
  return data;
}

// --- Studios ---

export async function getStudios(page = 0, pageSize = 20): Promise<PaginatedResponse<Studio>> {
  const { data } = await client.get<PaginatedResponse<Studio>>("/studios", {
    params: { page, pageSize },
  });
  return data;
}

export async function createStudio(studio: { name: string; email: string }): Promise<Studio> {
  const { data } = await client.post<Studio>("/studios", studio);
  return data;
}

export async function deleteStudio(id: number): Promise<void> {
  await client.delete(`/studios/${id}`);
}

// --- Users ---

export async function getUsers(page = 0, pageSize = 20): Promise<PaginatedResponse<User>> {
  const { data } = await client.get<PaginatedResponse<User>>("/users", {
    params: { page, pageSize },
  });
  return data;
}

export async function deleteUser(id: number): Promise<void> {
  await client.delete(`/users/${id}`);
}

// --- Events ---

export async function getEvents(page = 0, pageSize = 20): Promise<PaginatedResponse<Event>> {
  const { data } = await client.get<PaginatedResponse<Event>>("/events", {
    params: { page, pageSize },
  });
  return data;
}

// --- Subscriptions ---

export async function getSubscriptions(page = 0, pageSize = 20): Promise<PaginatedResponse<Subscription>> {
  const { data } = await client.get<PaginatedResponse<Subscription>>("/subscriptions", {
    params: { page, pageSize },
  });
  return data;
}

export async function createSubscription(sub: {
  studioId: number;
  plan: string;
  paymentMethod: "lahza" | "cash";
}): Promise<Subscription> {
  const { data } = await client.post<Subscription>("/subscriptions", sub);
  return data;
}
