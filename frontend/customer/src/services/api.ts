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

export interface EmbeddingCandidate {
  embeddingId: number;
  representativePhotoId: number | null;
  representativeFilename: string | null;
}

export interface EmbeddingSearchResult {
  candidates: EmbeddingCandidate[];
}

export interface MatchedPhoto {
  photoId: number;
  filename: string;
  url: string;
  similarityScore: number;
}

export interface OrderItem {
  id: number;
  orderId: number;
  photoIds: number[];
}

/** Validate an order ID and get order details including linked event. */
export async function getOrder(orderId: string): Promise<Order> {
  const { data } = await client.get<Order>(`/customer/orders/${orderId}`);
  return data;
}

/**
 * Step 1: Upload a self-photo for face search.
 * Returns top 10 embedding candidates for the customer to confirm.
 */
export async function searchEmbeddings(
  orderId: string,
  photoBlob: Blob,
): Promise<EmbeddingSearchResult> {
  const form = new FormData();
  form.append("photo", photoBlob, "selfie.jpg");
  const { data } = await client.post<EmbeddingSearchResult>(
    `/customer/orders/${orderId}/search`,
    form,
    { headers: { "Content-Type": "multipart/form-data" } },
  );
  return data;
}

/**
 * Step 2: Given confirmed embedding IDs, fetch all photos linked to them.
 */
export async function getPhotosByEmbeddings(
  orderId: string,
  embeddingIds: number[],
): Promise<MatchedPhoto[]> {
  const { data } = await client.post<{ matchedPhotos: MatchedPhoto[]; totalMatches: number }>(
    `/customer/orders/${orderId}/photos`,
    { embeddingIds },
  );
  return data.matchedPhotos;
}

/** Confirm selected photos for an order. */
export async function confirmOrder(
  orderId: string,
  photoIds: number[],
): Promise<void> {
  await client.post(`/customer/orders/${orderId}/confirm`, { photoIds });
}
