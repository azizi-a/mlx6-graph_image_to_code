import React from "react";
import { DocumentDuplicateIcon } from "@heroicons/react/24/outline";

export function CodeViewer({ code }: { code?: string }) {
  if (!code) return null;
  return (
    <div className="relative">
      <button
        onClick={() => {
          navigator.clipboard.writeText(code);
        }}
        className="absolute top-0 right-0 p-2 bg-gray-100 rounded-tl rounded-br dark:bg-gray-800 cursor-pointer">
        <DocumentDuplicateIcon className="h-5 w-5" />
      </button>
      <pre className="p-4 bg-gray-100 rounded max-w-full dark:bg-gray-800 dark:text-gray-100">
        <code className="whitespace-pre-wrap max-w-full">{code}</code>
      </pre>
    </div>
  );
}
