import axios from "axios";

const client = axios.create({
  baseURL: "/api",
  headers: { "Content-Type": "application/json" },
});

export interface Order {
  id: number;
  eventId: number;
  eventName: string;
  status: string;
  createdAt: string;
}

export interface MatchedPhoto {
  photoId: number;
  filename: string;
  url: string;
  similarity: number;
}

export interface OrderItem {
  id: number;
  orderId: number;
  photoIds: number[];
}

/** Validate an order ID and get order details including linked event. */
export async function getOrder(orderId: string): Promise<Order> {
  const { data } = await client.get<Order>(`/orders/${orderId}`);
  return data;
}

/**
 * Upload a self-photo for face search against the event linked to the order.
 * Returns matched photos sorted by similarity descending.
 */
export async function searchFaces(
  orderId: string,
  photoBlob: Blob,
): Promise<MatchedPhoto[]> {
  const form = new FormData();
  form.append("photo", photoBlob, "selfie.jpg");

  const { data } = await client.post<MatchedPhoto[]>(
    `/orders/${orderId}/search`,
    form,
    { headers: { "Content-Type": "multipart/form-data" } },
  );
  return data;
}

/** Confirm selected photos for an order. */
export async function confirmOrder(
  orderId: string,
  photoIds: number[],
): Promise<void> {
  await client.post(`/orders/${orderId}/confirm`, { photoIds });
}
