import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import { Pinecone } from "@pinecone-database/pinecone";
import { HfInference } from "@huggingface/inference";

dotenv.config();

console.log("PINECONE_API_KEY:", dotenv, process.env.PINECONE_API_KEY);


const app = express();
app.use(cors());
app.use(express.json());

const hf = new HfInference(process.env.HF_API_KEY);
const pinecone = new Pinecone({ apiKey: process.env.PINECONE_API_KEY });

const index = pinecone.index("chat-agent"); // Ensure the index name is correct

// 🔹 Function to retrieve relevant chunks from Pinecone
async function retrieveChunks(query) {
    try {
        console.log("🔍 Converting query to embedding...");
        const queryEmbedding = await hf.featureExtraction({
            model: "sentence-transformers/all-MiniLM-L6-v2",
            inputs: query,
        });

        if (!Array.isArray(queryEmbedding)) {
            throw new Error("🚨 Query embedding is not an array!");
        }

        console.log("🔍 Searching for relevant chunks in Pinecone...");
        const searchResults = await index.namespace("ns1").query({
            vector: queryEmbedding,
            topK: 3,
            includeMetadata: true,
        });

        if (!searchResults.matches || searchResults.matches.length === 0) {
            console.warn("⚠ No relevant chunks found!");
            return "";
        }

        return searchResults.matches.map((match) => match.metadata.text).join("\n");
    } catch (error) {
        console.error("❌ Error in retrieveChunks:", error);
        return "";
    }
}

// 🔹 Function to generate an answer using Hugging Face
async function generateAnswer(query) {
    try {
        console.log("\n🔹 Processing query:", query);
        const context = await retrieveChunks(query);

        if (!context) {
            return "⚠ No relevant data found.";
        }

        console.log("🔍 Generating answer using Hugging Face model...");
        const response = await hf.textGeneration({
           // model: "mistralai/Mistral-7B-Instruct-v0.3",
           // inputs: `Context: ${context} \n\n Question: ${query} \n\n Answer:`,
           // parameters: { max_new_tokens: 200, temperature: 0.3 },


           model: "mistralai/Mistral-7B-Instruct-v0.3",
           inputs: `You are an AI assistant. Answer the question briefly and concisely. Provide only one direct answer, without repeating or suggesting other questions. 
           Context: ${context} 
           Question: ${query} 
           Answer:`,
           parameters: { max_new_tokens: 200, temperature: 0.3, do_sample: false },

        });

        let generatedText = response.generated_text.split("Question:").join("\n\n\n\n  <br><br>Other Question:");

       // generatedText = generatedText.split("\n\n Question:")[0]; 


        const match = generatedText.match(/Answer:\s*(.*)/s);
        return match && match[1] ? match[1].trim() : generatedText;

    } catch (error) {
        console.error("❌ Error in generateAnswer:", error);
        return "❌ Error generating answer.";
    }
}

// 🔹 API Route for answering questions
app.post("/ask", async (req, res) => {
    const { question } = req.body;

    if (!question) {
        return res.status(400).json({ error: "Question is required!" });
    }

    const answer = await generateAnswer(question);
    res.json({ answer });
});

// 🔹 Start the server
const PORT = process.env.PORT || 5000;
app.listen(PORT, () => {
    console.log(`🚀 Server running on http://localhost:${PORT}`);
});
