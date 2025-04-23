"use client";

import React, { useState, useRef, useCallback } from "react";
import { CameraIcon, ArrowUpTrayIcon } from "@heroicons/react/24/outline";
import { LoadingWheel } from "../LoadingWheel";

interface ImageUploadProps {
  onImageSelected: (file: File) => void;
  loading?: boolean;
}

export const ImageUpload: React.FC<ImageUploadProps> = ({
  onImageSelected,
  loading,
}) => {
  const [isDragging, setIsDragging] = useState(false);
  const [previewImage, setPreviewImage] = useState<string | null>(null);
  const dropZoneRef = useRef<HTMLDivElement>(null);

  const handleImagePreviewed = useCallback(
    (file: File) => {
      if (file) {
        const reader = new FileReader();
        reader.onloadend = () => {
          setPreviewImage(reader.result as string);
        };
        reader.readAsDataURL(file);
      }
    },
    [onImageSelected]
  );

  const handleImageSelected = useCallback(() => {
    if (previewImage) {
      const blob = dataURItoBlob(previewImage);
      const file = new File([blob], "selected.png", {
        lastModified: Date.now(),
      });
      onImageSelected(file);
    }
  }, [onImageSelected, previewImage]);

  function dataURItoBlob(dataURI: string) {
    const byteString = atob(dataURI.split(",")[1]);
    const mimeString = dataURI.split(",")[0].split(":")[1].split(";")[0];
    const ab = new ArrayBuffer(byteString.length);
    const ia = new Uint8Array(ab);
    for (let i = 0; i < byteString.length; i++) {
      ia[i] = byteString.charCodeAt(i);
    }
    const blob = new Blob([ab], { type: mimeString });
    return blob;
  }

  const handleDrop = useCallback(
    (e: React.DragEvent<HTMLDivElement>) => {
      e.preventDefault();
      setIsDragging(false);

      if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        handleImagePreviewed(e.dataTransfer.files[0]);
      }
    },
    [handleImagePreviewed]
  );

  const handleDragOver = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleFileSelect = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      if (e.target.files && e.target.files[0]) {
        handleImagePreviewed(e.target.files[0]);
      }
    },
    [handleImagePreviewed]
  );

  const handleCameraSelect = useCallback(() => {
    navigator.mediaDevices
      .getUserMedia({ video: true })
      .then((stream) => {
        const video = document.createElement("video");
        video.srcObject = stream;
        video.play();

        // Create canvas for capturing image
        const canvas = document.createElement("canvas");
        const context = canvas.getContext("2d");

        // Capture image after 2 seconds
        setTimeout(() => {
          canvas.width = video.videoWidth;
          canvas.height = video.videoHeight;
          context?.drawImage(video, 0, 0);

          canvas.toBlob((blob) => {
            if (blob) {
              const file = new File([blob], "camera.jpg", {
                type: "image/jpeg",
              });
              handleImagePreviewed(file);
            }
          }, "image/jpeg");

          stream.getTracks().forEach((track) => track.stop());
        }, 2000);
      })
      .catch((error) => console.error("Camera error:", error));
  }, [handleImagePreviewed]);

  return (
    <div className="w-full max-w-md mx-auto">
      {previewImage ? (
        <div className="mb-4">
          <img
            src={previewImage}
            alt="Preview"
            className="w-full rounded-lg object-contain"
          />
          {!loading && (
            <div className="flex justify-around items-start mt-2">
              <div className="w-20" />
              <button
                onClick={handleImageSelected}
                className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 cursor-pointer">
                Submit
              </button>
              <button
                onClick={() => setPreviewImage(null)}
                className="text-xs font-medium text-red-600 hover:text-red-500 cursor-pointer">
                Remove image
              </button>
            </div>
          )}
          {loading && <LoadingWheel />}
        </div>
      ) : (
        <div
          ref={dropZoneRef}
          className={`border-2 border-dashed rounded-lg p-8 text-center transition-all duration-300 ${
            isDragging ? "border-blue-500 bg-blue-50" : "border-gray-300"
          }`}
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}>
          <div className="mb-4">
            <ArrowUpTrayIcon className="mx-auto h-10 w-10 text-gray-400" />
          </div>
          <div className="text-sm text-gray-600">
            <label className="relative cursor-pointer p-1 bg-white rounded-md font-medium text-indigo-600 hover:text-indigo-500 focus-within:outline-none focus-within:ring-2 focus-within:ring-offset-2 focus-within:ring-indigo-500">
              <span className="p-2 leading-loose">Upload a file</span>
              <input
                type="file"
                accept="image/*"
                className="sr-only"
                onChange={handleFileSelect}
              />
            </label>
            <p>or drag and drop</p>
          </div>
          <p className="text-xs text-gray-500">PNG, JPG, GIF up to 10MB</p>

          <button
            onClick={handleCameraSelect}
            className="mt-4 inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500">
            <CameraIcon className="-ml-1 mr-2 h-5 w-5" />
            Use Camera
          </button>
        </div>
      )}
    </div>
  );
};
