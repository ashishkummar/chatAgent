 
// API key for Pinecone (replace with actual key)
const API_KEY = "pcsk_3jcj9H_Nn5U7ixrNXy5eP98PN3e8AUXzRgS5XNSvwyPKUhmne9YGY19p7Pon5V1UnVvqq8"; 
  

// Pinecone Vector API URL
const VECTOR_API_URL = "https://chat-agent-fgvxm45.svc.aped-4627-b74a.pinecone.io/query";

  
// Generate a random vector with 384 dimensions
const queryVector = Array.from({ length: 384 }, () => Math.random());

// Fetch vector data from Pinecone
async function fetchVectorData() {
  try {
    const response = await fetch(VECTOR_API_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Api-Key": API_KEY, // Include API key if required
      },
      body: JSON.stringify({
        namespace: "ns1", // Correct namespace
        topK: 100, // Return top 100 results
        vector: queryVector, // Correct 384-dimensional vector
        includeMetadata: true,
        includeValues: true, // ✅ Include vector values in the response
      }),
    });

    if (!response.ok) {
      throw new Error(`Error: ${response.statusText}`);
    }

    const data = await response.json();
    console.log("API Response:", data); // Check if response is received
    return data.matches || []; // Extract matched vectors
  } catch (error) {
    console.error("Error fetching vector data:", error.message || error);
    return [];
  }
}

// Plot vector data using Plotly
async function plotVectorData() {
  const vectorData = await fetchVectorData();

  if (vectorData.length === 0) {
    console.warn("No vector data available.");
    return;
  }

  // Extract vector coordinates and metadata
  const xValues = vectorData.map((vec) => vec.values[0]);
  const yValues = vectorData.map((vec) => vec.values[1]);
  const zValues = vectorData.map((vec) => vec.values[2] || 0); // Fallback to 0 if no z-value

  // Plot with Plotly
  const trace = {
    x: xValues,
    y: yValues,
    z: zValues,
    mode: "markers",
    type: "scatter3d", // For 3D, use 'scatter' for 2D
    marker: {
      size: 6,
      color: zValues,
      colorscale: "Viridis",
      opacity: 0.8,
    },
    text: vectorData.map((vec) => vec.metadata.text), // Add text labels to points
  };

  const layout = {
    title: "3D Vector Data Visualization",
    margin: { l: 0, r: 0, b: 0, t: 40 },
    scene: {
      xaxis: { title: "X Axis" },
      yaxis: { title: "Y Axis" },
      zaxis: { title: "Z Axis" },
    },
  };

  Plotly.newPlot("vector-plot", [trace], layout);
}

// Run the plot function
plotVectorData();
