import React from "react";

export const LoadingWheel = () => {
  return (
    <div className="relative w-5 h-5">
      <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2">
        <div className="w-4 h-4 border-b-2 border-gray-900 rounded-full animate-spin"></div>
      </div>
    </div>
  );
};
