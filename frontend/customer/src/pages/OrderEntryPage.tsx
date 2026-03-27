import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { getOrder } from "../services/api";

export default function OrderEntryPage() {
  const navigate = useNavigate();
  const [orderId, setOrderId] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    const trimmed = orderId.trim();
    if (!trimmed) return;

    setError(null);
    setLoading(true);

    try {
      await getOrder(trimmed);
      navigate(`/camera/${trimmed}`);
    } catch {
      setError("Order not found. Please check your order ID and try again.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="flex flex-col items-center pt-12">
      <div className="w-full max-w-sm">
        <h2 className="text-2xl font-bold text-gray-900 text-center mb-2">
          Enter Your Order ID
        </h2>
        <p className="text-sm text-gray-500 text-center mb-8">
          You should have received an order ID from the studio after payment.
        </p>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label htmlFor="orderId" className="sr-only">
              Order ID
            </label>
            <input
              id="orderId"
              type="text"
              inputMode="numeric"
              pattern="[0-9]*"
              placeholder="e.g. 10042"
              value={orderId}
              onChange={(e) => setOrderId(e.target.value)}
              className="w-full rounded-lg border border-gray-300 px-4 py-3 text-center text-lg
                         focus:border-blue-500 focus:ring-2 focus:ring-blue-500 focus:outline-none"
              autoFocus
            />
          </div>

          {error && (
            <p className="text-sm text-red-600 text-center" role="alert">
              {error}
            </p>
          )}

          <button
            type="submit"
            disabled={loading || !orderId.trim()}
            className="w-full rounded-lg bg-blue-600 px-4 py-3 text-white font-medium
                       hover:bg-blue-700 focus:ring-2 focus:ring-blue-500 focus:ring-offset-2
                       disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            {loading ? "Verifying..." : "Continue"}
          </button>
        </form>
      </div>
    </div>
  );
}
