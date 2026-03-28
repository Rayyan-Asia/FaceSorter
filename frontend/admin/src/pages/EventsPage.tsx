import { useEffect, useState } from "react";
import { getEvents, type Event } from "../services/api";

export default function EventsPage() {
  const [events, setEvents] = useState<Event[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getEvents()
      .then((res) => setEvents(res))
      .catch(() => setError("Failed to load events."))
      .finally(() => setLoading(false));
  }, []);

  return (
    <div>
      <h2 className="text-2xl font-bold text-gray-900 mb-6">Events</h2>

      {error && <p className="text-red-600 mb-4">{error}</p>}

      {loading ? (
        <p className="text-gray-500">Loading...</p>
      ) : events.length === 0 ? (
        <p className="text-gray-500">No events created yet.</p>
      ) : (
        <div className="bg-white border border-gray-200 rounded-lg overflow-hidden">
          <table className="w-full text-sm text-left">
            <thead className="bg-gray-50 border-b border-gray-200">
              <tr>
                <th className="px-4 py-3 font-medium text-gray-500">ID</th>
                <th className="px-4 py-3 font-medium text-gray-500">Event Name</th>
                <th className="px-4 py-3 font-medium text-gray-500">Studio</th>
                <th className="px-4 py-3 font-medium text-gray-500">Photos</th>
                <th className="px-4 py-3 font-medium text-gray-500">Processing</th>
                <th className="px-4 py-3 font-medium text-gray-500">Created</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-100">
              {events.map((event) => {
                const progress =
                  event.photoCount > 0
                    ? Math.round((event.processedCount / event.photoCount) * 100)
                    : 0;

                return (
                  <tr key={event.id} className="hover:bg-gray-50">
                    <td className="px-4 py-3 text-gray-900">{event.id}</td>
                    <td className="px-4 py-3 font-medium text-gray-900">{event.name}</td>
                    <td className="px-4 py-3 text-gray-600">{event.studioName}</td>
                    <td className="px-4 py-3 text-gray-600">{event.photoCount.toLocaleString()}</td>
                    <td className="px-4 py-3">
                      <div className="flex items-center gap-2">
                        <div className="flex-1 h-2 bg-gray-200 rounded-full overflow-hidden max-w-[120px]">
                          <div
                            className="h-full bg-blue-600 rounded-full transition-all"
                            style={{ width: `${progress}%` }}
                          />
                        </div>
                        <span className="text-xs text-gray-500 whitespace-nowrap">
                          {event.processedCount}/{event.photoCount}
                        </span>
                      </div>
                    </td>
                    <td className="px-4 py-3 text-gray-500">
                      {new Date(event.createdAt).toLocaleDateString()}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
