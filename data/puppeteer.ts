import { spawn } from "child_process";
import * as path from "path";

// Number of graphs to generate (default: 5)
const numGraphs = process.argv[2] ? parseInt(process.argv[2]) : 5;

console.log(`Starting graph generator to create ${numGraphs} D3.js graphs...`);

// Run the graph generator
const graphGeneratorPath = path.join(
  process.cwd(),
  "dist",
  "graphGenerator.js"
);
const child = spawn("node", [graphGeneratorPath, numGraphs.toString()], {
  stdio: "inherit",
});

child.on("close", (code) => {
  if (code === 0) {
    console.log("Graph generation completed successfully!");
  } else {
    console.error(`Graph generation failed with code ${code}`);
  }
});
