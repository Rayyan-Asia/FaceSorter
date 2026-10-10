import { useEffect, useState, FormEvent } from "react";
import { register, getStudios, Studio } from "../services/api";

/**
 * Admin-only account provisioning. Accounts are never self-service: an authenticated admin
 * creates studio operators (and other admins). This does NOT sign in as the new account.
 */
export default function AccountsPage() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [role, setRole] = useState<"ADMIN" | "STUDIO_OPERATOR">("STUDIO_OPERATOR");
  const [studioId, setStudioId] = useState("");
  const [studios, setStudios] = useState<Studio[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [created, setCreated] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    getStudios().then(setStudios).catch(() => setStudios([]));
  }, []);

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    setError(null);
    setCreated(null);
    setLoading(true);
    try {
      const res = await register(
        email,
        password,
        role,
        role === "STUDIO_OPERATOR" ? Number(studioId) : undefined
      );
      setCreated(`${res.role === "ADMIN" ? "Admin" : "Studio operator"} account created for ${res.email}.`);
      setEmail("");
      setPassword("");
      setStudioId("");
    } catch (err: any) {
      const status = err.response?.status;
      if (status === 409) setError("That email is already registered.");
      else if (status === 400) setError("Check the details: password needs 8+ characters and operators need a studio.");
      else if (status === 403 || status === 401) setError("Only administrators can create accounts.");
      else setError("Could not create the account. Please try again.");
    } finally {
      setLoading(false);
    }
  }

  const input =
    "w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500";

  return (
    <div className="max-w-md">
      <h2 className="text-xl font-semibold text-gray-900 mb-1">Create account</h2>
      <p className="text-sm text-gray-500 mb-4">
        Studio operators sign in to the desktop app; admins sign in here.
      </p>

      <form onSubmit={handleSubmit} className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 space-y-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Email</label>
          <input type="email" value={email} onChange={(e) => setEmail(e.target.value)} required className={input} />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Password</label>
          <input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
            minLength={8}
            className={input}
            placeholder="Min. 8 characters"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Role</label>
          <select value={role} onChange={(e) => setRole(e.target.value as "ADMIN" | "STUDIO_OPERATOR")} className={input}>
            <option value="STUDIO_OPERATOR">Studio Operator</option>
            <option value="ADMIN">Admin</option>
          </select>
        </div>

        {role === "STUDIO_OPERATOR" && (
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Studio</label>
            <select value={studioId} onChange={(e) => setStudioId(e.target.value)} required className={input}>
              <option value="">Select a studio…</option>
              {studios.map((s) => (
                <option key={s.id} value={s.id}>
                  {s.name}
                </option>
              ))}
            </select>
          </div>
        )}

        {error && <p className="text-sm text-red-600" role="alert">{error}</p>}
        {created && <p className="text-sm text-green-700" role="status">{created}</p>}

        <button
          type="submit"
          disabled={loading}
          className="w-full rounded-lg bg-blue-600 px-4 py-2.5 text-sm font-medium text-white hover:bg-blue-700 disabled:opacity-50 transition-colors"
        >
          {loading ? "Creating account..." : "Create account"}
        </button>
      </form>
    </div>
  );
}
