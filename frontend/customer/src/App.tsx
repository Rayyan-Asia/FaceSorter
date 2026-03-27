import { Routes, Route, Navigate } from "react-router-dom";
import OrderEntryPage from "./pages/OrderEntryPage";
import CameraPage from "./pages/CameraPage";
import PhotoSelectPage from "./pages/PhotoSelectPage";

export default function App() {
  return (
    <div className="min-h-screen bg-gray-50">
      <header className="bg-white border-b border-gray-200 px-6 py-4">
        <h1 className="text-xl font-semibold text-gray-900">FaceSorter</h1>
        <p className="text-sm text-gray-500">Find your photos</p>
      </header>

      <main className="mx-auto max-w-2xl px-4 py-8">
        <Routes>
          <Route path="/" element={<OrderEntryPage />} />
          <Route path="/camera/:orderId" element={<CameraPage />} />
          <Route path="/photos/:orderId" element={<PhotoSelectPage />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </main>
    </div>
  );
}
