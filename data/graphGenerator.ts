import * as fs from "fs";
import * as path from "path";
import puppeteer from "puppeteer";

// Define graph types
type GraphType = "bar" | "line" | "pie" | "scatter" | "area" | "doughnut";

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

  // Generate random axis labels based on graph type
  const axisLabels = generateAxisLabels(type);
  const xAxisLabel = axisLabels.x;
  const yAxisLabel = axisLabels.y;

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
  .text("${xAxisLabel}");

g.append("text")
  .attr("class", "y-axis-label")
  .attr("transform", "rotate(-90)")
  .attr("x", -innerHeight / 2)
  .attr("y", -40)
  .style("text-anchor", "middle")
  .text("${yAxisLabel}");
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
  .text("${xAxisLabel}");

g.append("text")
  .attr("class", "y-axis-label")
  .attr("transform", "rotate(-90)")
  .attr("x", -innerHeight / 2)
  .attr("y", -40)
  .style("text-anchor", "middle")
  .text("${yAxisLabel}");
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
  .text("${xAxisLabel}");

g.append("text")
  .attr("class", "y-axis-label")
  .attr("transform", "rotate(-90)")
  .attr("x", -innerHeight / 2)
  .attr("y", -40)
  .style("text-anchor", "middle")
  .text("${yAxisLabel}");
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
  .attr("fill", "${colors[0]}")
  .attr("fill-opacity", 0.6)
  .attr("d", area);

g.append("path")
  .datum(data)
  .attr("fill", "none")
  .attr("stroke", "${colors[1]}")
  .attr("stroke-width", 2)
  .attr("d", line);

// Add labels
g.append("text")
  .attr("class", "x-axis-label")
  .attr("x", innerWidth / 2)
  .attr("y", innerHeight + 40)
  .style("text-anchor", "middle")
  .text("${xAxisLabel}");

g.append("text")
  .attr("class", "y-axis-label")
  .attr("transform", "rotate(-90)")
  .attr("x", -innerHeight / 2)
  .attr("y", -40)
  .style("text-anchor", "middle")
  .text("${yAxisLabel}");
      `;

    case "doughnut":
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
  .innerRadius(radius * 0.5)
  .outerRadius(radius);

const labelArc = d3.arc()
  .innerRadius(radius * 0.7)
  .outerRadius(radius * 0.7);

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

    default:
      return `console.error("Unsupported graph type: ${type}");`;
  }
}

// Function to generate random axis labels based on graph type
function generateAxisLabels(type: GraphType): { x: string; y: string } {
  // Common X-axis label options
  const xAxisOptions = [
    "Time",
    "Date",
    "Month",
    "Year",
    "Quarter",
    "Week",
    "Category",
    "Product",
    "Region",
    "Country",
    "Age Group",
    "Income Level",
    "Education Level",
    "Department",
    "Team",
    "Segment",
    "Channel",
    "Platform",
    "Device",
    "Version",
  ];

  // Common Y-axis label options
  const yAxisOptions = [
    "Value",
    "Amount",
    "Revenue",
    "Sales",
    "Profit",
    "Cost",
    "Count",
    "Frequency",
    "Percentage",
    "Growth Rate",
    "Conversion Rate",
    "Engagement",
    "Satisfaction",
    "Performance",
    "Efficiency",
    "Score",
    "Rating",
    "Index",
    "Utilization",
    "Adoption",
  ];

  // Specific options based on graph type
  let typeSpecificX: string[] = [];
  let typeSpecificY: string[] = [];

  switch (type) {
    case "bar":
      typeSpecificX = ["Category", "Product", "Region", "Department", "Group"];
      typeSpecificY = ["Value", "Amount", "Revenue", "Count", "Percentage"];
      break;
    case "line":
      typeSpecificX = ["Time", "Date", "Month", "Year", "Quarter", "Week"];
      typeSpecificY = ["Value", "Growth", "Trend", "Rate", "Performance"];
      break;
    case "scatter":
      typeSpecificX = [
        "Variable A",
        "Factor X",
        "Input",
        "Cause",
        "Dimension 1",
      ];
      typeSpecificY = [
        "Variable B",
        "Factor Y",
        "Output",
        "Effect",
        "Dimension 2",
      ];
      break;
    case "area":
      typeSpecificX = [
        "Time Period",
        "Date Range",
        "Timeline",
        "Epoch",
        "Phase",
      ];
      typeSpecificY = [
        "Volume",
        "Quantity",
        "Accumulation",
        "Total",
        "Aggregate",
      ];
      break;
    default:
      break;
  }

  // Combine common options with type-specific options
  const xOptions = [...new Set([...typeSpecificX, ...xAxisOptions])];
  const yOptions = [...new Set([...typeSpecificY, ...yAxisOptions])];

  // Randomly select labels
  const xLabel = xOptions[Math.floor(Math.random() * xOptions.length)];
  const yLabel = yOptions[Math.floor(Math.random() * yOptions.length)];

  return { x: xLabel, y: yLabel };
}

// Function to generate sample data for different graph types
function generateSampleData(type: GraphType): any[] {
  const length = Math.floor(Math.random() * 30 + 5);
  const xMagnitude = 10 ** Math.floor(Math.random() * 4);
  const yMagnitude = 10 ** Math.floor(Math.random() * 4);

  switch (type) {
    case "bar":
      const barData = generateBarChartData();
      return barData.labels.map((label, index) => ({
        label,
        value: barData.datasets[0].data[index],
      }));

    case "pie":
      const pieData = generatePieChartData();
      return pieData.labels.map((label, index) => ({
        label,
        value: pieData.datasets[0].data[index],
      }));

    case "line":
    case "area":
      return Array.from({ length }, (_, i) => ({
        x: i * xMagnitude,
        y: Math.floor(Math.random() * yMagnitude * 100) / 100,
      }));

    case "scatter":
      return Array.from({ length }, () => ({
        x: Math.floor(Math.random() * xMagnitude * 100) / 100, // Round to nearest integer
        y: Math.floor(Math.random() * yMagnitude * 100) / 100, // Round to nearest integer
        size: Math.ceil(Math.random() * 10), // Round to nearest integer
      }));

    case "doughnut":
      const doughnutData = generateDoughnutChartData();
      return doughnutData.labels.map((label, index) => ({
        label,
        value: doughnutData.datasets[0].data[index],
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

  const graphTypes: GraphType[] = [
    "bar",
    "line",
    "pie",
    "scatter",
    "area",
    "doughnut",
  ];

  // Launch browser
  const browser = await puppeteer.launch({
    args: ["--no-sandbox", "--disable-setuid-sandbox"],
  });

  for (let i = 0; i < count; i++) {
    // Select a random graph type
    const graphType = graphTypes[i % graphTypes.length];

    // Generate sample data
    const data = generateSampleData(graphType);

    // Generate complementary colors
    const colorScheme = generateComplementaryColors();

    // Create graph configuration
    const config: GraphConfig = {
      type: graphType,
      title: `${graphType.charAt(0).toUpperCase() + graphType.slice(1)} Chart ${
        i + 1
      }`,
      width: 800,
      height: 500,
      data,
      colors: colorScheme,
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

export function generateBarChartData() {
  // Large pool of potential labels
  const monthLabels = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
  ];

  const quarterLabels = ["Q1", "Q2", "Q3", "Q4"];

  const regionLabels = [
    "North America",
    "Europe",
    "Asia Pacific",
    "Latin America",
    "Middle East",
    "Africa",
    "Oceania",
    "Southeast Asia",
    "Eastern Europe",
    "Western Europe",
    "Central America",
    "Caribbean",
  ];

  const productLabels = [
    "Smartphones",
    "Laptops",
    "Tablets",
    "Wearables",
    "Accessories",
    "Software",
    "Services",
    "Cloud Storage",
    "Smart Home",
    "Gaming",
    "Audio",
    "Cameras",
    "Printers",
    "Networking",
    "Components",
  ];

  // Randomly choose which type of labels to use
  const labelTypes = [monthLabels, quarterLabels, regionLabels, productLabels];
  const selectedLabelSet =
    labelTypes[Math.floor(Math.random() * labelTypes.length)];

  // Select a random subset of labels (3-8)
  const shuffled = [...selectedLabelSet].sort(() => 0.5 - Math.random());
  const selectedLabels = shuffled.slice(0, Math.floor(Math.random() * 6) + 3);

  // Random dataset names
  const datasetOptions = [
    ["Revenue", "Expenses"],
    ["Sales", "Returns"],
    ["Actual", "Target"],
    ["2022", "2023"],
    ["Direct", "Indirect"],
    ["Online", "In-store"],
    ["Domestic", "International"],
  ];

  const selectedDataset =
    datasetOptions[Math.floor(Math.random() * datasetOptions.length)];

  // Generate truly random data with varying scales for each dataset
  const magnitude = 10 ** Math.floor(Math.random() * 5);
  const dataset1Max = Math.random() * magnitude * 10;
  const dataset2Max = Math.random() * magnitude * 10;

  return {
    labels: selectedLabels,
    datasets: [
      {
        label: selectedDataset[0],
        data: selectedLabels.map(
          () => Math.floor(Math.random() * dataset1Max * 10) / 10
        ),
        backgroundColor: "rgba(75, 192, 192, 0.6)",
        borderColor: "rgba(54, 162, 235, 1)",
        borderWidth: 1,
      },
      {
        label: selectedDataset[1],
        data: selectedLabels.map(
          () => Math.floor(Math.random() * dataset2Max * 10) / 10
        ),
        backgroundColor: "rgba(255, 99, 132, 0.6)",
        borderColor: "rgba(255, 99, 132, 1)",
        borderWidth: 1,
      },
    ],
  };
}

export function generateLineChartData() {
  // Large pool of potential labels
  const timeLabels = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
    "Week 1",
    "Week 2",
    "Week 3",
    "Week 4",
    "Week 5",
    "Week 6",
    "Day 1",
    "Day 2",
    "Day 3",
    "Day 4",
    "Day 5",
    "Day 6",
    "Day 7",
  ];

  const categoryLabels = [
    "Product A",
    "Product B",
    "Product C",
    "Product D",
    "Product E",
    "Category 1",
    "Category 2",
    "Category 3",
    "Category 4",
    "Category 5",
    "Team Alpha",
    "Team Beta",
    "Team Gamma",
    "Team Delta",
    "Team Epsilon",
    "Project X",
    "Project Y",
    "Project Z",
    "Project Omega",
    "Project Sigma",
  ];

  const metricLabels = [
    "10km",
    "20km",
    "30km",
    "40km",
    "50km",
    "60km",
    "100m",
    "200m",
    "300m",
    "400m",
    "500m",
    "Level 1",
    "Level 2",
    "Level 3",
    "Level 4",
    "Level 5",
  ];

  // Randomly choose which type of labels to use
  const labelTypes = [timeLabels, categoryLabels, metricLabels];
  const selectedLabelSet =
    labelTypes[Math.floor(Math.random() * labelTypes.length)];

  // Select a random subset of labels (4-10)
  const shuffled = [...selectedLabelSet].sort(() => 0.5 - Math.random());
  const selectedLabels = shuffled.slice(0, Math.floor(Math.random() * 7) + 4);

  // Random dataset names
  const datasetOptions = [
    ["Primary", "Secondary"],
    ["2022", "2023"],
    ["Group A", "Group B"],
    ["Morning", "Evening"],
    ["First Attempt", "Second Attempt"],
    ["Baseline", "Improved"],
    ["Control", "Test"],
  ];

  const selectedDataset =
    datasetOptions[Math.floor(Math.random() * datasetOptions.length)];

  return {
    labels: selectedLabels,
    datasets: [
      {
        label: selectedDataset[0],
        data: selectedLabels.map(() => Math.floor(Math.random() * 500) + 200),
        borderColor: "rgba(54, 162, 235, 1)",
        backgroundColor: "rgba(54, 162, 235, 0.2)",
        tension: 0.4,
      },
      {
        label: selectedDataset[1],
        data: selectedLabels.map(() => Math.floor(Math.random() * 600) + 300),
        borderColor: "rgba(255, 159, 64, 1)",
        backgroundColor: "rgba(255, 159, 64, 0.2)",
        tension: 0.4,
      },
    ],
  };
}

export function generatePieChartData() {
  // Multiple sets of potential labels
  const marketSegments = [
    "Enterprise",
    "Small Business",
    "Consumer",
    "Government",
    "Education",
    "Healthcare",
    "Retail",
    "Manufacturing",
    "Finance",
    "Technology",
    "Energy",
    "Transportation",
    "Media",
    "Hospitality",
  ];

  const ageGroups = [
    "18-24",
    "25-34",
    "35-44",
    "45-54",
    "55-64",
    "65+",
    "Gen Z",
    "Millennials",
    "Gen X",
    "Baby Boomers",
    "Silent Generation",
  ];

  const deviceTypes = [
    "Desktop",
    "Mobile",
    "Tablet",
    "Smart TV",
    "Game Console",
    "Wearable",
    "IoT Device",
    "Smart Speaker",
    "VR Headset",
  ];

  const foodCategories = [
    "Vegetables",
    "Fruits",
    "Grains",
    "Protein",
    "Dairy",
    "Snacks",
    "Beverages",
    "Desserts",
    "Fast Food",
    "Organic",
  ];

  const transportModes = [
    "Car",
    "Bus",
    "Train",
    "Bicycle",
    "Walking",
    "Rideshare",
    "Scooter",
    "Motorcycle",
    "Subway",
    "Ferry",
  ];

  // Randomly choose which type of labels to use
  const labelSets = [
    marketSegments,
    ageGroups,
    deviceTypes,
    foodCategories,
    transportModes,
  ];
  const selectedLabelSet =
    labelSets[Math.floor(Math.random() * labelSets.length)];

  // Select a random number of segments (3-7)
  const numSegments = Math.floor(Math.random() * 6) + 2;
  const selectedSegments = [...selectedLabelSet]
    .sort(() => 0.5 - Math.random())
    .slice(0, numSegments);

  // Generate completely random values (no constraints on summing to 100)
  // We'll normalize them later
  const rawData = selectedSegments.map(
    () => Math.floor(Math.random() * 95) + 5
  );

  // Random vibrant colors
  const generateColor = () => {
    const h = Math.floor(Math.random() * 360);
    return `hsl(${h}, 70%, 60%)`;
  };

  return {
    labels: selectedSegments,
    datasets: [
      {
        data: rawData,
        backgroundColor: selectedSegments.map(() => generateColor()),
      },
    ],
  };
}

export function generateDoughnutChartData() {
  // Multiple sets of potential labels
  const socialPlatforms = [
    "Facebook",
    "Instagram",
    "Twitter",
    "LinkedIn",
    "TikTok",
    "YouTube",
    "Pinterest",
    "Reddit",
    "Snapchat",
    "WhatsApp",
  ];

  const operatingSystems = [
    "Windows",
    "macOS",
    "iOS",
    "Android",
    "Linux",
    "Chrome OS",
    "watchOS",
    "tvOS",
    "Ubuntu",
    "Fedora",
  ];

  const programmingLanguages = [
    "JavaScript",
    "Python",
    "Java",
    "C#",
    "PHP",
    "TypeScript",
    "Ruby",
    "Swift",
    "Go",
    "Rust",
  ];

  const musicGenres = [
    "Pop",
    "Rock",
    "Hip Hop",
    "R&B",
    "Country",
    "Electronic",
    "Jazz",
    "Classical",
    "Folk",
    "Metal",
  ];

  const sportTypes = [
    "Football",
    "Basketball",
    "Baseball",
    "Soccer",
    "Tennis",
    "Golf",
    "Hockey",
    "Swimming",
    "Volleyball",
    "Athletics",
  ];

  // Randomly choose which type of labels to use
  const labelSets = [
    socialPlatforms,
    operatingSystems,
    programmingLanguages,
    musicGenres,
    sportTypes,
  ];
  const selectedLabelSet =
    labelSets[Math.floor(Math.random() * labelSets.length)];

  // Select a random number of types (2-6)
  const numTypes = Math.floor(Math.random() * 5) + 2;
  const selectedTypes = [...selectedLabelSet]
    .sort(() => 0.5 - Math.random())
    .slice(0, numTypes);

  // Generate completely random values with no constraints
  const rawData = selectedTypes.map(() => Math.floor(Math.random() * 95) + 5);

  // Generate random vibrant colors
  const generateColor = () => {
    const h = Math.floor(Math.random() * 360);
    const s = Math.floor(Math.random() * 30) + 70;
    const l = Math.floor(Math.random() * 20) + 50;
    return `hsla(${h}, ${s}%, ${l}%, 0.7)`;
  };

  return {
    labels: selectedTypes,
    datasets: [
      {
        data: rawData,
        backgroundColor: selectedTypes.map(() => generateColor()),
      },
    ],
  };
}

// Function to generate complementary color schemes
function generateComplementaryColors(): string[] {
  // Predefined color schemes
  const colorSchemes = [
    // Blue and orange theme
    ["#4e79a7", "#f28e2c", "#76b7b2", "#e15759", "#59a14f"],
    // Purple and green theme
    ["#9c755f", "#bab0ab", "#59a14f", "#76b7b2", "#e15759"],
    // Teal and red theme
    ["#76b7b2", "#e15759", "#f28e2c", "#4e79a7", "#59a14f"],
    // Green and purple theme
    ["#59a14f", "#af7aa1", "#ff9da7", "#9c755f", "#bab0ab"],
    // Blue and yellow theme
    ["#4e79a7", "#edc949", "#76b7b2", "#ff9da7", "#9c755f"],
    // Red and blue theme
    ["#e15759", "#4e79a7", "#f28e2c", "#76b7b2", "#59a14f"],
    // Orange and teal theme
    ["#f28e2c", "#76b7b2", "#e15759", "#59a14f", "#4e79a7"],
    // Purple and orange theme
    ["#b07aa1", "#f28e2c", "#4e79a7", "#59a14f", "#76b7b2"],
    // Green and red theme
    ["#59a14f", "#e15759", "#f28e2c", "#76b7b2", "#4e79a7"],
    // Blue and green theme
    ["#4e79a7", "#59a14f", "#f28e2c", "#e15759", "#76b7b2"],
  ];

  // Select a random color scheme
  return colorSchemes[Math.floor(Math.random() * colorSchemes.length)];
}
