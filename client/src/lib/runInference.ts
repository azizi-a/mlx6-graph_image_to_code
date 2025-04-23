export async function runInference(imageBlob: Blob): Promise<string> {
  const fd = new FormData();
  fd.append("file", imageBlob, "upload.png");

  const res = await fetch("http://localhost:8000/generate", {
    method: "POST",
    body: fd,
  });

  if (!res.ok) {
    throw new Error(await res.text());
  }

  const payload = await res.json();
  return payload.code;
}
