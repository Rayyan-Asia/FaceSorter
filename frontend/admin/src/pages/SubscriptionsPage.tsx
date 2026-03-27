import { useEffect, useState } from "react";
import {
  getSubscriptions,
  createSubscription,
  getStudios,
  type Subscription,
  type Studio,
} from "../services/api";

export default function SubscriptionsPage() {
  const [subscriptions, setSubscriptions] = useState<Subscription[]>([]);
  const [studios, setStudios] = useState<Studio[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [showForm, setShowForm] = useState(false);
  const [formStudioId, setFormStudioId] = useState("");
  const [formPlan, setFormPlan] = useState("annual");
  const [formPayment, setFormPayment] = useState<"lahza" | "cash">("cash");
  const [formError, setFormError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  function load() {
    setLoading(true);
    Promise.all([getSubscriptions(), getStudios()])
      .then(([subRes, studioRes]) => {
        setSubscriptions(subRes.items);
        setStudios(studioRes.items);
      })
      .catch(() => setError("Failed to load data."))
      .finally(() => setLoading(false));
  }

  useEffect(() => { load(); }, []);

  async function handleCreate(e: React.FormEvent) {
    e.preventDefault();
    if (!formStudioId) return;

    setSubmitting(true);
    setFormError(null);

    try {
      await createSubscription({
        studioId: Number(formStudioId),
        plan: formPlan,
        paymentMethod: formPayment,
      });
      setShowForm(false);
      setFormStudioId("");
      load();
    } catch {
      setFormError("Failed to create subscription.");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-2xl font-bold text-gray-900">Subscriptions</h2>
        <button
          onClick={() => setShowForm(!showForm)}
          className="rounded-lg bg-blue-600 px-4 py-2 text-sm text-white font-medium hover:bg-blue-700 transition-colors"
        >
          {showForm ? "Cancel" : "Add Subscription"}
        </button>
      </div>

      {showForm && (
        <form onSubmit={handleCreate} className="bg-white border border-gray-200 rounded-lg p-4 mb-6 space-y-3">
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
            <select
              value={formStudioId}
              onChange={(e) => setFormStudioId(e.target.value)}
              className="rounded-md border border-gray-300 px-3 py-2 text-sm focus:border-blue-500 focus:ring-1 focus:ring-blue-500 focus:outline-none"
              required
            >
              <option value="">Select studio...</option>
              {studios.map((s) => (
                <option key={s.id} value={s.id}>
                  {s.name}
                </option>
              ))}
            </select>

            <select
              value={formPlan}
              onChange={(e) => setFormPlan(e.target.value)}
              className="rounded-md border border-gray-300 px-3 py-2 text-sm focus:border-blue-500 focus:ring-1 focus:ring-blue-500 focus:outline-none"
            >
              <option value="annual">Annual</option>
            </select>

            <select
              value={formPayment}
              onChange={(e) => setFormPayment(e.target.value as "lahza" | "cash")}
              className="rounded-md border border-gray-300 px-3 py-2 text-sm focus:border-blue-500 focus:ring-1 focus:ring-blue-500 focus:outline-none"
            >
              <option value="cash">Cash</option>
              <option value="lahza">Lahza</option>
            </select>
          </div>
          {formError && <p className="text-sm text-red-600">{formError}</p>}
          <button
            type="submit"
            disabled={submitting}
            className="rounded-md bg-blue-600 px-4 py-2 text-sm text-white font-medium hover:bg-blue-700 disabled:opacity-50 transition-colors"
          >
            {submitting ? "Creating..." : "Create Subscription"}
          </button>
        </form>
      )}

      {error && <p className="text-red-600 mb-4">{error}</p>}

      {loading ? (
        <p className="text-gray-500">Loading...</p>
      ) : subscriptions.length === 0 ? (
        <p className="text-gray-500">No subscriptions yet.</p>
      ) : (
        <div className="bg-white border border-gray-200 rounded-lg overflow-hidden">
          <table className="w-full text-sm text-left">
            <thead className="bg-gray-50 border-b border-gray-200">
              <tr>
                <th className="px-4 py-3 font-medium text-gray-500">ID</th>
                <th className="px-4 py-3 font-medium text-gray-500">Studio</th>
                <th className="px-4 py-3 font-medium text-gray-500">Plan</th>
                <th className="px-4 py-3 font-medium text-gray-500">Status</th>
                <th className="px-4 py-3 font-medium text-gray-500">Payment</th>
                <th className="px-4 py-3 font-medium text-gray-500">Start</th>
                <th className="px-4 py-3 font-medium text-gray-500">End</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-100">
              {subscriptions.map((sub) => (
                <tr key={sub.id} className="hover:bg-gray-50">
                  <td className="px-4 py-3 text-gray-900">{sub.id}</td>
                  <td className="px-4 py-3 font-medium text-gray-900">{sub.studioName}</td>
                  <td className="px-4 py-3 text-gray-600 capitalize">{sub.plan}</td>
                  <td className="px-4 py-3">
                    <StatusBadge status={sub.status} />
                  </td>
                  <td className="px-4 py-3 text-gray-600 capitalize">{sub.paymentMethod}</td>
                  <td className="px-4 py-3 text-gray-500">{new Date(sub.startDate).toLocaleDateString()}</td>
                  <td className="px-4 py-3 text-gray-500">{new Date(sub.endDate).toLocaleDateString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

function StatusBadge({ status }: { status: string }) {
  const styles: Record<string, string> = {
    active: "bg-green-100 text-green-800",
    expired: "bg-red-100 text-red-800",
    pending: "bg-yellow-100 text-yellow-800",
  };

  return (
    <span className={`inline-block rounded-full px-2.5 py-0.5 text-xs font-medium ${styles[status] ?? "bg-gray-100 text-gray-600"}`}>
      {status}
    </span>
  );
}
