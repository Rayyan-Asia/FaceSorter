import { useEffect, useState } from "react";
import { getStudios, createStudio, deleteStudio, type Studio } from "../services/api";

export default function StudiosPage() {
  const [studios, setStudios] = useState<Studio[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showForm, setShowForm] = useState(false);
  const [formName, setFormName] = useState("");
  const [formEmail, setFormEmail] = useState("");
  const [formError, setFormError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  function load() {
    setLoading(true);
    getStudios()
      .then((res) => setStudios(res.items))
      .catch(() => setError("Failed to load studios."))
      .finally(() => setLoading(false));
  }

  useEffect(() => { load(); }, []);

  async function handleCreate(e: React.FormEvent) {
    e.preventDefault();
    if (!formName.trim() || !formEmail.trim()) return;

    setSubmitting(true);
    setFormError(null);

    try {
      await createStudio({ name: formName.trim(), email: formEmail.trim() });
      setFormName("");
      setFormEmail("");
      setShowForm(false);
      load();
    } catch {
      setFormError("Failed to create studio.");
    } finally {
      setSubmitting(false);
    }
  }

  async function handleDelete(id: number) {
    if (!window.confirm("Delete this studio? This cannot be undone.")) return;
    try {
      await deleteStudio(id);
      setStudios((prev) => prev.filter((s) => s.id !== id));
    } catch {
      setError("Failed to delete studio.");
    }
  }

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-2xl font-bold text-gray-900">Studios</h2>
        <button
          onClick={() => setShowForm(!showForm)}
          className="rounded-lg bg-blue-600 px-4 py-2 text-sm text-white font-medium hover:bg-blue-700 transition-colors"
        >
          {showForm ? "Cancel" : "Add Studio"}
        </button>
      </div>

      {showForm && (
        <form onSubmit={handleCreate} className="bg-white border border-gray-200 rounded-lg p-4 mb-6 space-y-3">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            <input
              type="text"
              placeholder="Studio name"
              value={formName}
              onChange={(e) => setFormName(e.target.value)}
              className="rounded-md border border-gray-300 px-3 py-2 text-sm focus:border-blue-500 focus:ring-1 focus:ring-blue-500 focus:outline-none"
              required
            />
            <input
              type="email"
              placeholder="Contact email"
              value={formEmail}
              onChange={(e) => setFormEmail(e.target.value)}
              className="rounded-md border border-gray-300 px-3 py-2 text-sm focus:border-blue-500 focus:ring-1 focus:ring-blue-500 focus:outline-none"
              required
            />
          </div>
          {formError && <p className="text-sm text-red-600">{formError}</p>}
          <button
            type="submit"
            disabled={submitting}
            className="rounded-md bg-blue-600 px-4 py-2 text-sm text-white font-medium hover:bg-blue-700 disabled:opacity-50 transition-colors"
          >
            {submitting ? "Creating..." : "Create Studio"}
          </button>
        </form>
      )}

      {error && <p className="text-red-600 mb-4">{error}</p>}

      {loading ? (
        <p className="text-gray-500">Loading...</p>
      ) : studios.length === 0 ? (
        <p className="text-gray-500">No studios registered yet.</p>
      ) : (
        <div className="bg-white border border-gray-200 rounded-lg overflow-hidden">
          <table className="w-full text-sm text-left">
            <thead className="bg-gray-50 border-b border-gray-200">
              <tr>
                <th className="px-4 py-3 font-medium text-gray-500">ID</th>
                <th className="px-4 py-3 font-medium text-gray-500">Name</th>
                <th className="px-4 py-3 font-medium text-gray-500">Email</th>
                <th className="px-4 py-3 font-medium text-gray-500">Subscription</th>
                <th className="px-4 py-3 font-medium text-gray-500">Created</th>
                <th className="px-4 py-3 font-medium text-gray-500" />
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-100">
              {studios.map((studio) => (
                <tr key={studio.id} className="hover:bg-gray-50">
                  <td className="px-4 py-3 text-gray-900">{studio.id}</td>
                  <td className="px-4 py-3 font-medium text-gray-900">{studio.name}</td>
                  <td className="px-4 py-3 text-gray-600">{studio.email}</td>
                  <td className="px-4 py-3">
                    <StatusBadge status={studio.subscriptionStatus} />
                  </td>
                  <td className="px-4 py-3 text-gray-500">{new Date(studio.createdAt).toLocaleDateString()}</td>
                  <td className="px-4 py-3 text-right">
                    <button
                      onClick={() => handleDelete(studio.id)}
                      className="text-red-600 hover:underline text-sm"
                    >
                      Delete
                    </button>
                  </td>
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
    none: "bg-gray-100 text-gray-600",
  };

  return (
    <span className={`inline-block rounded-full px-2.5 py-0.5 text-xs font-medium ${styles[status] ?? styles["none"]}`}>
      {status}
    </span>
  );
}
