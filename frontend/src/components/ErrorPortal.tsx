import React from 'react';
import { AlertTriangle, RefreshCw } from 'lucide-react';

interface ErrorPortalProps {
  title?: string;
  message: string;
  onRetry?: () => void;
}

export const ErrorPortal: React.FC<ErrorPortalProps> = ({
  title = 'Portal Connection Interrupted',
  message,
  onRetry,
}) => {
  return (
    <div className="mx-auto my-8 max-w-xl rounded-2xl border border-rose-500/40 bg-rose-950/30 p-6 text-center backdrop-blur-md">
      <div className="mx-auto mb-3 flex h-12 w-12 items-center justify-center rounded-full bg-rose-500/20 text-rose-400">
        <AlertTriangle className="h-6 w-6" />
      </div>
      <h3 className="font-fantasy text-lg font-bold text-rose-200">{title}</h3>
      <p className="mt-2 text-sm text-rose-200/80">{message}</p>
      {onRetry && (
        <button
          onClick={onRetry}
          className="mt-4 inline-flex items-center gap-2 rounded-lg border border-rose-400/40 bg-rose-500/20 px-4 py-2 text-xs font-semibold text-rose-100 transition hover:bg-rose-500/35"
        >
          <RefreshCw className="h-3.5 w-3.5" />
          Re-establish Connection
        </button>
      )}
    </div>
  );
};

