import { Pinecone } from "@pinecone-database/pinecone";
import { HfInference } from "@huggingface/inference";
import dotenv from "dotenv";

dotenv.config();
const hf = new HfInference(process.env.HF_API_KEY);
const pinecone = new Pinecone({ apiKey: process.env.PINECONE_API_KEY });

const index = pinecone.index("chat-agent"); // ✅ Make sure this matches your actual index name

const indexInfo = await index.describeIndexStats();
console.log("📊 Pinecone Index Stats:", JSON.stringify(indexInfo, null, 2));

async function retrieveChunks(query) {
  try {
    console.log("🔍 Converting query to embedding...");
    const queryEmbedding = await hf.featureExtraction({
      model: "sentence-transformers/all-MiniLM-L6-v2",
      inputs: query,
    });

    console.log("✅ Query embedding generated:", queryEmbedding.length);

    if (!Array.isArray(queryEmbedding)) {
      throw new Error("🚨 Query embedding is not an array!");
    }

    console.log("🔍 Searching for relevant chunks in Pinecone...");
    const searchResults = await index.namespace("ns1").query({
        vector: queryEmbedding,
        topK: 3,
        includeMetadata: true,
      });
      

    console.log("✅ Pinecone search results:", searchResults);

    if (!searchResults.matches || searchResults.matches.length === 0) {
      console.warn("⚠ No relevant chunks found!");
      return "";
    }

    const context = searchResults.matches.map((match) => match.metadata.text).join("\n");
   /// console.log("Retrieved Context:\n", context);
    return context;
  } catch (error) {
    console.error("❌ Error in retrieveChunks:", error);
    return "";
  }
}

async function generateAnswer(query) {
  
    try {
        console.log("\n🔹 Processing query:", query);
        const context = await retrieveChunks(query);
    
        if (!context) {
          console.warn("⚠ No context retrieved, skipping answer generation.");
          return;
        }
    
        console.log("🔍 Generating answer using Hugging Face model...");
        const response = await hf.textGeneration({
          model: "mistralai/Mistral-7B-Instruct-v0.3",
          inputs: `Context: ${context} \n\n Question: ${query} \n\n Answer:`,
          parameters: { max_new_tokens: 200, temperature: 0.3 },
        });
    
        // Extract only the answer part
        const generatedText = response.generated_text;
        const answerIndex = generatedText.indexOf("Answer:");
        
        if (answerIndex !== -1) {
          console.log("\n💬 Chatbot Answer:\n", generatedText.substring(answerIndex));
        } else {
          console.log("\n💬 Chatbot Answer:\n", generatedText);
        }
    
      } catch (error) {
        console.error("❌ Error in generateAnswer:", error);
      }



}

// ✅ Run the script properly
(async () => {
  try {
    
    
    const userQuery = "tell me about Multiple Video Support  avalibity ";  // ask here ------------ 🦸


    await generateAnswer(userQuery);
  } catch (error) {
    console.error("❌ Unexpected error:", error);
  }
}


)();
