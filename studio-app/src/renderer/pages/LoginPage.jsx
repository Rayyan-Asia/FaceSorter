import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuthStore } from '../store/authStore';
import { authApi } from '../services/api';
import Spinner from '../components/Spinner';

export default function LoginPage() {
  const navigate = useNavigate();
  const setAuth = useAuthStore((s) => s.setAuth);
  const isAuthenticated = useAuthStore((s) => s.isAuthenticated);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  if (isAuthenticated) {
    navigate('/events', { replace: true });
    return null;
  }

  const handleGoogleLogin = async () => {
    setLoading(true);
    setError(null);
    try {
      // In production, this would use Google OAuth popup via Electron.
      // For now, open the backend's OAuth endpoint and handle the redirect.
      const width = 500;
      const height = 600;
      const left = window.screenX + (window.outerWidth - width) / 2;
      const top = window.screenY + (window.outerHeight - height) / 2;

      const popup = window.open(
        'http://localhost:8080/api/auth/google/login',
        'Google Login',
        `width=${width},height=${height},left=${left},top=${top}`
      );

      // Listen for the OAuth callback message
      const handleMessage = async (event) => {
        if (event.data?.type === 'oauth-callback' && event.data.credential) {
          window.removeEventListener('message', handleMessage);
          popup?.close();

          try {
            const result = await authApi.googleLogin(event.data.credential);
            setAuth(result.token, result.user);
            navigate('/events', { replace: true });
          } catch (err) {
            setError('Authentication failed. Please try again.');
          }
        }
      };

      window.addEventListener('message', handleMessage);

      // Fallback: check if popup was blocked
      if (!popup) {
        window.removeEventListener('message', handleMessage);
        setError('Popup blocked. Please allow popups for this application.');
      }
    } catch (err) {
      setError('Could not initiate login. Is the backend running?');
    } finally {
      setLoading(false);
    }
  };

  // Dev-mode bypass for testing without a backend
  const handleDevLogin = () => {
    setAuth('dev-token', { name: 'Studio Operator', email: 'operator@studio.local' });
    navigate('/events', { replace: true });
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-100">
      <div className="card w-full max-w-sm text-center">
        <h1 className="text-2xl font-bold mb-2">FaceSorter</h1>
        <p className="text-sm text-gray-500 mb-8">Studio Operator Portal</p>

        {error && (
          <div className="mb-4 p-3 bg-red-50 text-red-700 text-sm rounded-lg">{error}</div>
        )}

        <button
          onClick={handleGoogleLogin}
          disabled={loading}
          className="w-full flex items-center justify-center gap-3 bg-white border border-gray-300 rounded-lg px-4 py-3 text-sm font-medium text-gray-700 hover:bg-gray-50 transition-colors disabled:opacity-50"
        >
          {loading ? (
            <Spinner size="sm" />
          ) : (
            <svg className="w-5 h-5" viewBox="0 0 24 24">
              <path
                d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92a5.06 5.06 0 0 1-2.2 3.32v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.1z"
                fill="#4285F4"
              />
              <path
                d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"
                fill="#34A853"
              />
              <path
                d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"
                fill="#FBBC05"
              />
              <path
                d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"
                fill="#EA4335"
              />
            </svg>
          )}
          Sign in with Google
        </button>

        <div className="mt-4 pt-4 border-t border-gray-200">
          <button
            onClick={handleDevLogin}
            className="text-xs text-gray-400 hover:text-gray-600 transition-colors"
          >
            Dev mode (skip auth)
          </button>
        </div>
      </div>
    </div>
  );
}
