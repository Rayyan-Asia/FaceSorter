import axios from "axios";

const client = axios.create({
  baseURL: "/api",
  headers: { "Content-Type": "application/json" },
});

// Attach JWT to every request if present
client.interceptors.request.use((config) => {
  const token = localStorage.getItem("fs_token");
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// Redirect to login on 401
client.interceptors.response.use(
  (r) => r,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem("fs_token");
      localStorage.removeItem("fs_user");
      window.location.href = "/login";
    }
    return Promise.reject(error);
  }
);

// --- Auth types ---

export interface AuthUser {
  email: string;
  role: string;
  studioId: number | null;
}

export interface AuthResponse {
  token: string;
  email: string;
  role: string;
  studioId: number | null;
}

// --- Auth ---

export async function login(email: string, password: string): Promise<AuthResponse> {
  const { data } = await client.post<AuthResponse>("/auth/login", { email, password });
  return data;
}

export async function register(
  email: string,
  password: string,
  role: "ADMIN" | "STUDIO_OPERATOR",
  studioId?: number
): Promise<AuthResponse> {
  const { data } = await client.post<AuthResponse>("/auth/register", {
    email,
    password,
    role,
    studioId: studioId ?? null,
  });
  return data;
}

// --- Types ---

export interface Studio {
  id: number;
  name: string;
  email: string;
  active: boolean;
  subscriptionStatus: string;
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
  status: string;
  startDate: string;
  endDate: string;
  paymentMethod: string;
  amountPaid: number;
  active: boolean;
}

export interface DashboardStats {
  totalStudios: number;
  totalUsers: number;
  totalEvents: number;
  activeSubscriptions: number;
  totalOrders: number;
}

// --- Dashboard ---

export async function getDashboardStats(): Promise<DashboardStats> {
  const { data } = await client.get<DashboardStats>("/admin/dashboard/stats");
  return data;
}

// --- Studios ---

export async function getStudios(): Promise<Studio[]> {
  const { data } = await client.get<Studio[]>("/admin/studios");
  return data;
}

export async function createStudio(studio: { name: string; email: string }): Promise<Studio> {
  const { data } = await client.post<Studio>("/admin/studios", studio);
  return data;
}

export async function deleteStudio(id: number): Promise<void> {
  await client.delete(`/admin/studios/${id}`);
}

// --- Users ---

export async function getUsers(): Promise<User[]> {
  const { data } = await client.get<User[]>("/admin/users");
  return data;
}

export async function deleteUser(id: number): Promise<void> {
  await client.delete(`/admin/users/${id}`);
}

// --- Events ---

export async function getEvents(): Promise<Event[]> {
  const { data } = await client.get<Event[]>("/admin/events");
  return data;
}

// --- Subscriptions ---

export async function getSubscriptions(): Promise<Subscription[]> {
  const { data } = await client.get<Subscription[]>("/admin/subscriptions");
  return data;
}

export async function createSubscription(sub: {
  studioId: number;
  plan?: string;
  startDate?: string;
  endDate?: string;
  paymentMethod: string;
  amountPaid?: number;
}): Promise<Subscription> {
  const { data } = await client.post<Subscription>("/admin/subscriptions", sub);
  return data;
}
