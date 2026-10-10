import { create } from 'zustand';

const TOKEN_KEY = 'auth_token';
const USER_KEY = 'auth_user';
// Treat a token as expired slightly early so we never send one that dies in flight.
const EXPIRY_SKEW_MS = 5000;
// setTimeout overflows (fires immediately) above 2^31-1 ms.
const MAX_TIMER_MS = 2 ** 31 - 1;

/** Returns the token's expiry as epoch ms, or null if it can't be read. */
function getTokenExpiryMs(token) {
  try {
    const payload = token.split('.')[1];
    const json = atob(payload.replace(/-/g, '+').replace(/_/g, '/'));
    const { exp } = JSON.parse(json);
    return typeof exp === 'number' ? exp * 1000 : null;
  } catch {
    return null;
  }
}

/** A token we cannot parse is treated as expired so a corrupt value can't wedge the app. */
export function isTokenExpired(token) {
  const expiryMs = getTokenExpiryMs(token);
  return expiryMs === null || expiryMs - EXPIRY_SKEW_MS <= Date.now();
}

function readStoredSession() {
  const token = localStorage.getItem(TOKEN_KEY);
  if (!token) return { token: null, user: null };
  if (isTokenExpired(token)) {
    localStorage.removeItem(TOKEN_KEY);
    localStorage.removeItem(USER_KEY);
    return { token: null, user: null };
  }
  try {
    return { token, user: JSON.parse(localStorage.getItem(USER_KEY) || 'null') };
  } catch {
    return { token, user: null };
  }
}

let logoutTimer = null;

export const useAuthStore = create((set, get) => {
  const clearTimer = () => {
    if (logoutTimer) clearTimeout(logoutTimer);
    logoutTimer = null;
  };

  // Log the user out the moment the token expires, even if they are idle.
  const scheduleExpiryLogout = (token) => {
    clearTimer();
    const expiryMs = getTokenExpiryMs(token);
    if (expiryMs === null) return;
    const delay = Math.min(Math.max(expiryMs - EXPIRY_SKEW_MS - Date.now(), 0), MAX_TIMER_MS);
    logoutTimer = setTimeout(() => get().logout(), delay);
  };

  const session = readStoredSession();
  if (session.token) scheduleExpiryLogout(session.token);

  return {
    token: session.token,
    user: session.user,
    isAuthenticated: !!session.token,

    setAuth: (token, user) => {
      localStorage.setItem(TOKEN_KEY, token);
      localStorage.setItem(USER_KEY, JSON.stringify(user));
      scheduleExpiryLogout(token);
      set({ token, user, isAuthenticated: true });
    },

    logout: () => {
      clearTimer();
      localStorage.removeItem(TOKEN_KEY);
      localStorage.removeItem(USER_KEY);
      set({ token: null, user: null, isAuthenticated: false });
    },
  };
});
