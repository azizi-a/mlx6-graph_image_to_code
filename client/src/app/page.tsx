import Image from "next/image";

import Inference from "@/components/Inference";

export default function Home() {
  return (
    <div className="grid grid-rows-[1rem_1fr_1rem] items-center justify-items-center min-h-screen p-8 pb-20 gap-16 sm:p-20 font-[family-name:var(--font-roboto-sans)]">
      <main className="grid-cols-12 gap-1 row-start-1 row-span-2 items-center text-center sm:items-start">
        <h1 className="text-4xl font-bold sm:text-5xl">
          D3.js code from images
        </h1>
        <div className="mt-4" />
        <p className="text-xl font-semibold">
          Upload an image of a chart or graph and get the D3 code to recreate
          it.
        </p>

        <div className="mt-16">
          <h2 className="text-2xl font-bold text-center sm:text-4xl">
            Upload an image
          </h2>
          <div className="m-2">
            <Inference />
          </div>
        </div>
      </main>

      <footer className="row-start-3 flex gap-[24px] flex-wrap items-center justify-center">
        <a
          className="flex items-center gap-2 hover:underline hover:underline-offset-4"
          href="https://github.com/azizi-a/mlx6-graph_image_to_code"
          target="_blank"
          rel="noopener noreferrer">
          <Image
            aria-hidden
            src="/github.svg"
            alt="GitHub icon"
            width={20}
            height={20}
          />
          See this project on GitHub
        </a>
        <a
          className="flex items-center gap-2 hover:underline hover:underline-offset-4"
          href="https://d3js.org/"
          target="_blank"
          rel="noopener noreferrer">
          <Image
            aria-hidden
            src="/globe.svg"
            alt="Globe icon"
            width={16}
            height={16}
          />
          Go to d3js.org →
        </a>
      </footer>
    </div>
  );
}
