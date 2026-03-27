import { Routes, Route, NavLink, Navigate } from "react-router-dom";
import DashboardPage from "./pages/DashboardPage";
import StudiosPage from "./pages/StudiosPage";
import UsersPage from "./pages/UsersPage";
import EventsPage from "./pages/EventsPage";
import SubscriptionsPage from "./pages/SubscriptionsPage";

const navItems = [
  { to: "/", label: "Dashboard" },
  { to: "/studios", label: "Studios" },
  { to: "/users", label: "Users" },
  { to: "/events", label: "Events" },
  { to: "/subscriptions", label: "Subscriptions" },
] as const;

export default function App() {
  return (
    <div className="min-h-screen bg-gray-50 flex">
      <aside className="w-56 bg-gray-900 text-white flex flex-col shrink-0">
        <div className="px-5 py-5 border-b border-gray-700">
          <h1 className="text-lg font-semibold">FaceSorter</h1>
          <p className="text-xs text-gray-400">Admin Panel</p>
        </div>
        <nav className="flex-1 px-3 py-4 space-y-1">
          {navItems.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.to === "/"}
              className={({ isActive }) =>
                `block px-3 py-2 rounded-md text-sm font-medium transition-colors ${
                  isActive
                    ? "bg-gray-800 text-white"
                    : "text-gray-300 hover:bg-gray-800 hover:text-white"
                }`
              }
            >
              {item.label}
            </NavLink>
          ))}
        </nav>
      </aside>

      <div className="flex-1 flex flex-col">
        <header className="bg-white border-b border-gray-200 px-6 py-4 flex items-center justify-between">
          <h2 className="text-lg font-semibold text-gray-900">Admin</h2>
        </header>

        <main className="flex-1 p-6">
          <Routes>
            <Route path="/" element={<DashboardPage />} />
            <Route path="/studios" element={<StudiosPage />} />
            <Route path="/users" element={<UsersPage />} />
            <Route path="/events" element={<EventsPage />} />
            <Route path="/subscriptions" element={<SubscriptionsPage />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </main>
      </div>
    </div>
  );
}
