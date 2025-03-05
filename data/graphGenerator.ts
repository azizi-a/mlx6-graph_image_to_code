import * as fs from "fs";
import * as path from "path";
import puppeteer from "puppeteer";

// Define graph types
type GraphType = "bar" | "line" | "pie" | "scatter" | "area";

interface GraphConfig {
  type: GraphType;
  title: string;
  width: number;
  height: number;
  data: any[];
  colors?: string[];
}

// Function to generate D3.js code for different graph types
function generateD3Code(config: GraphConfig): string {
  const {
    type,
    title,
    width,
    height,
    data,
    colors = ["steelblue", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"],
  } = config;

  // Common HTML template
  const htmlTemplate = `
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>${title}</title>
  <script src="https://d3js.org/d3.v7.min.js"></script>
  <style>
    body { font-family: Arial, sans-serif; }
    .chart-container { margin: 20px; }
  </style>
</head>
<body>
  <div class="chart-container">
    <h2>${title}</h2>
    <svg id="chart" width="${width}" height="${height}"></svg>
  </div>
  <script>
  // D3.js code
  ${generateGraphSpecificCode(type, data, width, height, colors)}
  </script>
</body>
</html>
  `;

  return htmlTemplate;
}

// Function to generate specific D3 code based on graph type
function generateGraphSpecificCode(
  type: GraphType,
  data: any[],
  width: number,
  height: number,
  colors: string[]
): string {
  const margin = { top: 40, right: 30, bottom: 60, left: 60 };
  const innerWidth = width - margin.left - margin.right;
  const innerHeight = height - margin.top - margin.bottom;

  switch (type) {
    case "bar":
      return `
const svg = d3.select("#chart");
const margin = { top: 40, right: 30, bottom: 60, left: 60 };
const width = ${width};
const height = ${height};
const innerWidth = width - margin.left - margin.right;
const innerHeight = height - margin.top - margin.bottom;

const data = ${JSON.stringify(data)};

const x = d3.scaleBand()
  .domain(data.map(d => d.label))
  .range([0, innerWidth])
  .padding(0.1);

const y = d3.scaleLinear()
  .domain([0, d3.max(data, d => d.value)])
  .nice()
  .range([innerHeight, 0]);

const g = svg.append("g")
  .attr("transform", \`translate(\${margin.left},\${margin.top})\`);

g.append("g")
  .attr("class", "x-axis")
  .attr("transform", \`translate(0,\${innerHeight})\`)
  .call(d3.axisBottom(x))
  .selectAll("text")
  .attr("transform", "rotate(-45)")
  .style("text-anchor", "end");

g.append("g")
  .attr("class", "y-axis")
  .call(d3.axisLeft(y));

g.selectAll(".bar")
  .data(data)
  .enter().append("rect")
  .attr("class", "bar")
  .attr("x", d => x(d.label))
  .attr("y", d => y(d.value))
  .attr("width", x.bandwidth())
  .attr("height", d => innerHeight - y(d.value))
  .attr("fill", "${colors[0]}");

// Add labels
g.append("text")
  .attr("class", "x-axis-label")
  .attr("x", innerWidth / 2)
  .attr("y", innerHeight + 40)
  .style("text-anchor", "middle")
  .text("Categories");

g.append("text")
  .attr("class", "y-axis-label")
  .attr("transform", "rotate(-90)")
  .attr("x", -innerHeight / 2)
  .attr("y", -40)
  .style("text-anchor", "middle")
  .text("Values");
      `;

    case "line":
      return `
const svg = d3.select("#chart");
const margin = { top: 40, right: 30, bottom: 60, left: 60 };
const width = ${width};
const height = ${height};
const innerWidth = width - margin.left - margin.right;
const innerHeight = height - margin.top - margin.bottom;

const data = ${JSON.stringify(data)};

const x = d3.scaleLinear()
  .domain(d3.extent(data, d => d.x))
  .range([0, innerWidth]);

const y = d3.scaleLinear()
  .domain([0, d3.max(data, d => d.y)])
  .nice()
  .range([innerHeight, 0]);

const line = d3.line()
  .x(d => x(d.x))
  .y(d => y(d.y))
  .curve(d3.curveMonotoneX);

const g = svg.append("g")
  .attr("transform", \`translate(\${margin.left},\${margin.top})\`);

g.append("g")
  .attr("class", "x-axis")
  .attr("transform", \`translate(0,\${innerHeight})\`)
  .call(d3.axisBottom(x));

g.append("g")
  .attr("class", "y-axis")
  .call(d3.axisLeft(y));

g.append("path")
  .datum(data)
  .attr("fill", "none")
  .attr("stroke", "${colors[0]}")
  .attr("stroke-width", 2)
  .attr("d", line);

// Add dots
g.selectAll(".dot")
  .data(data)
  .enter().append("circle")
  .attr("class", "dot")
  .attr("cx", d => x(d.x))
  .attr("cy", d => y(d.y))
  .attr("r", 4)
  .attr("fill", "${colors[1]}");

// Add labels
g.append("text")
  .attr("class", "x-axis-label")
  .attr("x", innerWidth / 2)
  .attr("y", innerHeight + 40)
  .style("text-anchor", "middle")
  .text("X Axis");

g.append("text")
  .attr("class", "y-axis-label")
  .attr("transform", "rotate(-90)")
  .attr("x", -innerHeight / 2)
  .attr("y", -40)
  .style("text-anchor", "middle")
  .text("Y Axis");
      `;

    case "pie":
      return `
const svg = d3.select("#chart");
const width = ${width};
const height = ${height};
const radius = Math.min(width, height) / 2 - 40;

const data = ${JSON.stringify(data)};

const color = d3.scaleOrdinal()
  .domain(data.map(d => d.label))
  .range(${JSON.stringify(colors)});

const pie = d3.pie()
  .value(d => d.value)
  .sort(null);

const arc = d3.arc()
  .innerRadius(0)
  .outerRadius(radius);

const labelArc = d3.arc()
  .innerRadius(radius * 0.6)
  .outerRadius(radius * 0.6);

const g = svg.append("g")
  .attr("transform", \`translate(\${width / 2},\${height / 2})\`);

const arcs = g.selectAll(".arc")
  .data(pie(data))
  .enter().append("g")
  .attr("class", "arc");

arcs.append("path")
  .attr("d", arc)
  .attr("fill", d => color(d.data.label))
  .attr("stroke", "white")
  .style("stroke-width", "2px");

arcs.append("text")
  .attr("transform", d => \`translate(\${labelArc.centroid(d)})\`)
  .attr("dy", ".35em")
  .style("text-anchor", "middle")
  .style("font-size", "12px")
  .text(d => d.data.label);

// Add legend
const legend = svg.append("g")
  .attr("transform", \`translate(\${width - 120}, 20)\`);

data.forEach((d, i) => {
  const legendRow = legend.append("g")
    .attr("transform", \`translate(0, \${i * 20})\`);
    
  legendRow.append("rect")
    .attr("width", 10)
    .attr("height", 10)
    .attr("fill", color(d.label));
    
  legendRow.append("text")
    .attr("x", 15)
    .attr("y", 10)
    .attr("text-anchor", "start")
    .style("font-size", "12px")
    .text(d.label);
});
      `;

    case "scatter":
      return `
const svg = d3.select("#chart");
const margin = { top: 40, right: 30, bottom: 60, left: 60 };
const width = ${width};
const height = ${height};
const innerWidth = width - margin.left - margin.right;
const innerHeight = height - margin.top - margin.bottom;

const data = ${JSON.stringify(data)};

const x = d3.scaleLinear()
  .domain([0, d3.max(data, d => d.x)])
  .nice()
  .range([0, innerWidth]);

const y = d3.scaleLinear()
  .domain([0, d3.max(data, d => d.y)])
  .nice()
  .range([innerHeight, 0]);

const g = svg.append("g")
  .attr("transform", \`translate(\${margin.left},\${margin.top})\`);

g.append("g")
  .attr("class", "x-axis")
  .attr("transform", \`translate(0,\${innerHeight})\`)
  .call(d3.axisBottom(x));

g.append("g")
  .attr("class", "y-axis")
  .call(d3.axisLeft(y));

g.selectAll(".dot")
  .data(data)
  .enter().append("circle")
  .attr("class", "dot")
  .attr("cx", d => x(d.x))
  .attr("cy", d => y(d.y))
  .attr("r", d => d.size || 5)
  .attr("fill", d => d.color || "${colors[0]}")
  .attr("opacity", 0.7);

// Add labels
g.append("text")
  .attr("class", "x-axis-label")
  .attr("x", innerWidth / 2)
  .attr("y", innerHeight + 40)
  .style("text-anchor", "middle")
  .text("X Axis");

g.append("text")
  .attr("class", "y-axis-label")
  .attr("transform", "rotate(-90)")
  .attr("x", -innerHeight / 2)
  .attr("y", -40)
  .style("text-anchor", "middle")
  .text("Y Axis");
      `;

    case "area":
      return `
const svg = d3.select("#chart");
const margin = { top: 40, right: 30, bottom: 60, left: 60 };
const width = ${width};
const height = ${height};
const innerWidth = width - margin.left - margin.right;
const innerHeight = height - margin.top - margin.bottom;

const data = ${JSON.stringify(data)};

const x = d3.scaleLinear()
  .domain(d3.extent(data, d => d.x))
  .range([0, innerWidth]);

const y = d3.scaleLinear()
  .domain([0, d3.max(data, d => d.y)])
  .nice()
  .range([innerHeight, 0]);

const area = d3.area()
  .x(d => x(d.x))
  .y0(innerHeight)
  .y1(d => y(d.y))
  .curve(d3.curveMonotoneX);

const g = svg.append("g")
  .attr("transform", \`translate(\${margin.left},\${margin.top})\`);

g.append("g")
  .attr("class", "x-axis")
  .attr("transform", \`translate(0,\${innerHeight})\`)
  .call(d3.axisBottom(x));

g.append("g")
  .attr("class", "y-axis")
  .call(d3.axisLeft(y));

g.append("path")
  .datum(data)
  .attr("fill", "${colors[0]}")
  .attr("fill-opacity", 0.6)
  .attr("stroke", "${colors[1]}")
  .attr("stroke-width", 2)
  .attr("d", area);

// Add labels
g.append("text")
  .attr("class", "x-axis-label")
  .attr("x", innerWidth / 2)
  .attr("y", innerHeight + 40)
  .style("text-anchor", "middle")
  .text("X Axis");

g.append("text")
  .attr("class", "y-axis-label")
  .attr("transform", "rotate(-90)")
  .attr("x", -innerHeight / 2)
  .attr("y", -40)
  .style("text-anchor", "middle")
  .text("Y Axis");
      `;

    default:
      return `console.error("Unsupported graph type: ${type}");`;
  }
}

// Generate sample data for different graph types
function generateSampleData(type: GraphType): any[] {
  switch (type) {
    case "bar":
      return [
        { label: "A", value: 10 },
        { label: "B", value: 20 },
        { label: "C", value: 30 },
        { label: "D", value: 25 },
        { label: "E", value: 15 },
      ];

    case "line":
    case "area":
      return Array.from({ length: 10 }, (_, i) => ({
        x: i,
        y: Math.random() * 50 + 10,
      }));

    case "pie":
      return [
        { label: "Category A", value: 30 },
        { label: "Category B", value: 20 },
        { label: "Category C", value: 15 },
        { label: "Category D", value: 25 },
        { label: "Category E", value: 10 },
      ];

    case "scatter":
      return Array.from({ length: 30 }, () => ({
        x: Math.random() * 100,
        y: Math.random() * 100,
        size: Math.random() * 8 + 3,
      }));

    default:
      return [];
  }
}

// Function to generate HTML files with D3.js graphs
async function generateGraphs(count: number): Promise<void> {
  const outputDir = path.join(process.cwd(), "data", "generated_graphs");
  const codeDir = path.join(process.cwd(), "data", "generated_code");

  // Create directories if they don't exist
  if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
  }

  if (!fs.existsSync(codeDir)) {
    fs.mkdirSync(codeDir, { recursive: true });
  }

  const graphTypes: GraphType[] = ["bar", "line", "pie", "scatter", "area"];

  // Launch browser
  const browser = await puppeteer.launch({
    args: ["--no-sandbox", "--disable-setuid-sandbox"],
  });

  for (let i = 0; i < count; i++) {
    // Select a random graph type
    const graphType = graphTypes[i % graphTypes.length];

    // Generate sample data
    const data = generateSampleData(graphType);

    // Create graph configuration
    const config: GraphConfig = {
      type: graphType,
      title: `${graphType.charAt(0).toUpperCase() + graphType.slice(1)} Chart ${
        i + 1
      }`,
      width: 800,
      height: 500,
      data,
      colors: ["#4e79a7", "#f28e2c", "#e15759", "#76b7b2", "#59a14f"],
    };

    // Generate D3.js code
    const htmlContent = generateD3Code(config);

    // Save HTML file
    const htmlFilePath = path.join(
      outputDir,
      `graph_${i + 1}_${graphType}.html`
    );
    fs.writeFileSync(htmlFilePath, htmlContent);

    // Save D3.js code separately
    const codeFilePath = path.join(codeDir, `graph_${i + 1}_${graphType}.js`);
    const codeContent = generateGraphSpecificCode(
      graphType,
      data,
      config.width,
      config.height,
      config.colors || []
    );
    fs.writeFileSync(codeFilePath, codeContent);

    // Generate screenshot
    const page = await browser.newPage();
    await page.setContent(htmlContent);
    await page.setViewport({
      width: config.width + 100, // Add extra width for margins
      height: config.height + 100, // Add extra height for margins
    });

    const screenshotPath = path.join(
      outputDir,
      `graph_${i + 1}_${graphType}.png`
    );
    await page.screenshot({ path: screenshotPath });
    await page.close();

    console.log(`Generated graph ${i + 1}: ${config.title}`);
  }

  await browser.close();
  console.log(`Successfully generated ${count} D3.js graphs!`);
}

// Main function
async function main(): Promise<void> {
  const numGraphs = process.argv[2] ? parseInt(process.argv[2]) : 5;

  console.log(`Generating ${numGraphs} D3.js graphs...`);
  await generateGraphs(numGraphs);
}

main().catch((error) => {
  console.error("Error generating graphs:", error);
  process.exit(1);
});
