"use client";
import { useState, useTransition } from "react";
import { ImageUpload } from "@/components/ImageUpload";
import { CodeViewer } from "@/components/CodeViewer";
import { runInference } from "@/lib/runInference";

export default function Inference() {
  const [code, setCode] = useState<string>();
  const [isPending, start] = useTransition();
  const handleImage = (file: File) => {
    try {
      start(async () => setCode(await runInference(file)));
    } catch (error) {
      console.error("Error processing image:", error);
    }
  };

  return (
    <>
      <ImageUpload onImageSelected={handleImage} loading={isPending} />
      <div className="m-8">
        {code && (
          <>
            <div dangerouslySetInnerHTML={{ __html: code }} />
            <CodeViewer code={code} />
          </>
        )}
      </div>
    </>
  );
}
