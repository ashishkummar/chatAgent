import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import fs from "fs";
import mammoth from "mammoth";
import { HfInference } from "@huggingface/inference";
import dotenv from "dotenv";
import { Pinecone } from "@pinecone-database/pinecone";

dotenv.config();

/// Function to Extract Text from .docx
const extractText = async (filePath) => {
  const buffer = fs.readFileSync(filePath);
  const result = await mammoth.extractRawText({ buffer });
  return result.value;
};

const text = await extractText("data/Doc1.docx");
console.log("Extracted Text:", text.substring(0, 500)); // Preview first 500 chars

/// Splitting Text into Chunks
const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 500,
  chunkOverlap: 50,
});

const chunks = await splitter.createDocuments([text]);
console.log(`✅ Total Chunks: ${chunks.length}`);

//// EMBEDDINGS

const hf = new HfInference(process.env.HF_API_KEY);

const embedText = async (text) => {
  try {
    const response = await hf.featureExtraction({
      model: "sentence-transformers/all-MiniLM-L6-v2",
      inputs: text,
    });

    if (!Array.isArray(response) || response.length === 0) {
      throw new Error("Embedding returned empty array");
    }

    return response; // Returns embedding vector
  } catch (error) {
    console.error("❌ Error generating embedding:", error);
    return null; // Return null for invalid embeddings
  }
};

// Convert all chunks to embeddings
const embeddings = await Promise.all(chunks.map((chunk) => embedText(chunk.pageContent)));

// ✅ Filter out any null embeddings
const validEmbeddings = embeddings
  .map((emb, i) => (emb ? { emb, chunk: chunks[i] } : null))
  .filter(Boolean);

console.log(`✅ Valid Embeddings: ${validEmbeddings.length}`);

/////// PINECONE_

const pinecone = new Pinecone({
  apiKey: process.env.PINECONE_API_KEY,
});

async function ensureIndex() {
  try {
    const indexList = await pinecone.listIndexes();
    if (!indexList.indexes.some((idx) => idx.name === "chat-agent")) {
      await pinecone.createIndex({
        name: "chat-agent",
        dimension: 384, // Ensure it matches your model
        metric: "cosine",
        spec: {
          serverless: { cloud: "aws", region: "us-east-1" },
        },
      });
      console.log("✅ Pinecone Index Created with Dimension 384!");
    } else {
      console.log("✅ Pinecone Index already exists.");
    }
  } catch (error) {
    console.error("❌ Error ensuring index:", error);
  }
}

await ensureIndex();

/// Inserting Embeddings into Pinecone

const index = pinecone.index("chat-agent");

async function insertEmbeddings() {
  const index = pinecone.index("chat-agent"); // Ensure correct index name

  if (!Array.isArray(embeddings) || embeddings.length === 0) {
    console.error("❌ No valid embeddings found.");
    return;
  }

  if (!Array.isArray(chunks) || chunks.length !== embeddings.length) {
    console.error(`❌ Mismatch: Chunks (${chunks.length}) and Embeddings (${embeddings.length})`);
    return;
  }

  const vectors = chunks.map((chunk, i) => ({
    id: `chunk-${i}`,
    values: embeddings[i] || [], // Ensure `values` is always an array
    metadata: { text: chunk.pageContent },
  }));

  // ✅ Debugging Logs
  console.log("📌 Type of vectors:", typeof vectors);
  console.log("📌 Is vectors an array?", Array.isArray(vectors));
  console.log("📌 Length of vectors:", vectors.length);
  console.log("📌 First vector sample:", JSON.stringify(vectors[0], null, 2));

  try {
    if (!Array.isArray(vectors) || vectors.length === 0) {
      throw new TypeError("❌ Error: 'vectors' is not an array or is empty!");
    }

    // ✅ Use namespace before upsert
    const response = await index.namespace("ns1").upsert(vectors);

    console.log("✅ Successfully inserted into Pinecone!", response);
  } catch (error) {
    console.error("❌ Error inserting embeddings:", error);
  }
}


await insertEmbeddings();



const response = await index.namespace("ns1").query({
  topK: 2,
  vector: embeddings[0], // Ensure this is a valid 384-dimension vector
  includeValues: true,
  includeMetadata: true,
  // filter: { genre: { '$eq': 'action' }}  // ❌ REMOVE THIS if genre is not stored
});

console.log("🔍 Query Results:", JSON.stringify(response, null, 2));
