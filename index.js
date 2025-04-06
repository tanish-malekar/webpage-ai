import axios from "axios";
import * as cheerio from "cheerio";
import { ChromaClient } from "chromadb";
import { v4 as uuidv4 } from "uuid";
import { CohereClient } from 'cohere-ai';
import dotenv from 'dotenv';
import { COLLECTION_NAME } from "./constants.js";
import readline from 'readline';

dotenv.config();

async function scrapeParagraphDataFromWebPage(url) {
  const data = (await axios.get(url)).data;
  const $ = cheerio.load(data);

  const paragraphs = [];
  $("p").each((_, el) => {
    const text = $(el)
      .text()     // Get the text content
      .replace(/\[\d+\]/g, '')  // Remove [number] citations
      .trim();    // Remove whitespace
    
    if (text.length > 0) paragraphs.push(text);
  });

  return paragraphs;
}


const cohere = new CohereClient();

async function generateVectorEmbedding(text, inputType) {
  const embed = await cohere.v2.embed({
    texts: [text],
    model: 'embed-english-v3.0',
    inputType: inputType,
    embeddingTypes: ['float'],
  });
  return embed.embeddings.float[0]
}
  

const chromaClient = new ChromaClient({ path: "http://localhost:8000" });
chromaClient.heartbeat();

async function ingestEmbeddingToChromaDB(embedding, text) {
  const collection = await chromaClient.getOrCreateCollection({
    name: COLLECTION_NAME,
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
  const paragraphs = await scrapeParagraphDataFromWebPage(url);
  for (const paragraph of paragraphs) {
    const embedding = await generateVectorEmbedding(paragraph, "search_document");
    await ingestEmbeddingToChromaDB(embedding, paragraph);
  }
  console.log("Data scraped and ingested successfully");
}

async function ask(question){
  const collection = await chromaClient.getOrCreateCollection({
    name: COLLECTION_NAME,
  });
  const queryEmbedding = await generateVectorEmbedding(question, "search_query");
  const results = await collection.query({
    queryEmbeddings: [queryEmbedding],
    nResults: 3,
  });


  const response = await cohere.v2.chat({
    model: 'command-a-03-2025',
    messages: [
      {
        role: 'system',
        content: 'You are a helpful assistant who will answer the user\'s query by taking help of the retrieved context you have been provided with. The retrived context contains top 3 results in the form of JSON.',
      },
      {
        role: 'user',
        content: `Query: ${question} \n\n Context: ${JSON.stringify(results.metadatas[0])}`,
      },
    ],
  });

  return response.message.content[0].text
}

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});

async function processUserQuestions() {
  while (true) {
    try {
      const question = await new Promise((resolve) => {
        rl.question('Enter your question (or type "exit" to quit): ', resolve);
      });

      if (question.toLowerCase() === 'exit') {
        rl.close();
        break;
      }

      const response = await ask(question);
      console.log('\nAnswer:', response, '\n');
    } catch (error) {
      console.error('Error:', error.message);
    }
  }
}


// scrapAndIngest("https://en.wikipedia.org/wiki/Elon_Musk")
processUserQuestions();

