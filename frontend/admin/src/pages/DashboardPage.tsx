import { useEffect, useState } from "react";
import { getDashboardStats, type DashboardStats } from "../services/api";

export default function DashboardPage() {
  const [stats, setStats] = useState<DashboardStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getDashboardStats()
      .then(setStats)
      .catch(() => setError("Failed to load dashboard stats."))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return <p className="text-gray-500">Loading dashboard...</p>;
  }

  if (error) {
    return <p className="text-red-600">{error}</p>;
  }

  if (!stats) return null;

  const cards = [
    { label: "Studios", value: stats.totalStudios },
    { label: "Users", value: stats.totalUsers },
    { label: "Events", value: stats.totalEvents },
    { label: "Active Subscriptions", value: stats.activeSubscriptions },
    { label: "Total Orders", value: stats.totalOrders },
  ];

  return (
    <div>
      <h2 className="text-2xl font-bold text-gray-900 mb-6">Dashboard</h2>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5 gap-4">
        {cards.map((card) => (
          <div
            key={card.label}
            className="bg-white rounded-lg border border-gray-200 p-5"
          >
            <p className="text-sm text-gray-500 mb-1">{card.label}</p>
            <p className="text-3xl font-bold text-gray-900">{card.value.toLocaleString()}</p>
          </div>
        ))}
      </div>
    </div>
  );
}
