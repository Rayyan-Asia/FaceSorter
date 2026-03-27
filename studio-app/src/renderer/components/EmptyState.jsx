import React from 'react';

export default function EmptyState({ title, description, action }) {
  return (
    <div className="text-center py-12">
      <h3 className="text-sm font-medium text-gray-900">{title}</h3>
      {description && <p className="mt-1 text-sm text-gray-500">{description}</p>}
      {action && <div className="mt-4">{action}</div>}
    </div>
  );
}
