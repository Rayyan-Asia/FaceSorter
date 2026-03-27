import React from 'react';

export default function Spinner({ size = 'md' }) {
  const sizeClass = size === 'sm' ? 'h-4 w-4' : size === 'lg' ? 'h-8 w-8' : 'h-6 w-6';
  return (
    <div
      className={`${sizeClass} animate-spin rounded-full border-2 border-gray-300 border-t-primary-600`}
      role="status"
    >
      <span className="sr-only">Loading...</span>
    </div>
  );
}
