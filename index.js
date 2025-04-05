import axios from "axios";
import * as cheerio from "cheerio";
import { ChromaClient } from "chromadb";
import { pipeline } from "@xenova/transformers";
import { v4 as uuidv4 } from "uuid";

const extractor = await pipeline(
  "feature-extraction",
  "Xenova/all-MiniLM-L6-v2"
);
const chromaClient = new ChromaClient({ path: "http://localhost:8000" });
chromaClient.heartbeat();

async function scrapeWebPage(url) {
  const data = (await axios.get(url)).data;
  const $ = cheerio.load(data);

  const paragraphs = [];
  $("p").each((_, el) => {
    const text = $(el).text().trim();
    if (text.length > 0) paragraphs.push(text);
  });
  return paragraphs;
}

async function generateVectorEmbeddings(text) {
    const tokenEmbeddings = await extractor(text); // shape: [num_tokens][384]
  
    if (!Array.isArray(tokenEmbeddings) || tokenEmbeddings.length === 0) {
      throw new Error("Invalid or empty embedding output.");
    }
  
    const embeddingSize = tokenEmbeddings[0].length;
    const meanEmbedding = Array(embeddingSize).fill(0);
  
    for (const token of tokenEmbeddings) {
      for (let i = 0; i < embeddingSize; i++) {
        meanEmbedding[i] += token[i];
      }
    }
  
    for (let i = 0; i < embeddingSize; i++) {
      meanEmbedding[i] /= tokenEmbeddings.length;
    }
  
    return meanEmbedding;
  }
  

async function ingestToChromaDB(embedding, text) {
  const collection = await chromaClient.getOrCreateCollection({
    name: "scraped_data_v3",
  });
  console.log(embedding);
  const document = {
    ids: [uuidv4()],
    embeddings: [embedding],
    metadatas: { text: text },
  };
  await collection.add(document);
}

async function scrapAndIngest(url) {
  const paragraphs = await scrapeWebPage(url);
  for (const paragraph of paragraphs) {
    const embedding = await generateVectorEmbeddings(paragraph);
    await ingestToChromaDB(embedding, paragraph);
  }
  console.log("Data scraped and ingested successfully");
}


// console.log(await generateVectorEmbeddings("Hello world!"));
scrapAndIngest("https://en.wikipedia.org/wiki/Elon_Musk");
