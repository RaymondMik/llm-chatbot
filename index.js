import { OpenAI } from 'langchain/llms/openai';
import { ConversationalRetrievalQAChain } from 'langchain/chains';
import { HNSWLib } from 'langchain/vectorstores/hnswlib';
import { OpenAIEmbeddings } from 'langchain/embeddings/openai';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { BufferMemory } from 'langchain/memory';
import { PdfReader } from "pdfreader";
import chalk from 'chalk';
import * as fs from 'fs';
import readline from 'readline';
import dotenv from 'dotenv';

dotenv.config();

const log = console.log;
const openAIApiKey = process.env.OPENAI_API_KEY;
const rl = readline.createInterface({
   input: process.stdin,
   output: process.stdout
});

const readPdf = (userPrompt) => {
   let pdfText = '';
   fs.readFile('data/YOUR_PDF_FILE.pdf', (err, pdfBuffer) => {
      if (!err) {
         new PdfReader().parseBuffer(pdfBuffer, (err, item) => {
            if (err) console.error('error:', err);
            // when reading is completed
            else if (!item) {
               run(userPrompt, pdfText);
            }
            else if (item.text) {
               pdfText += ' ' + item.text;
            }
         });
        
      }
   })
}

const run = async (userPrompt, pdfText) => {
   /* Initialize the LLM to use to answer the question */
   const model = new OpenAI({ openAIApiKey, temperature: 0.9 });
   /* Load in the file we want to do question answering over */
   /* Split the text into chunks */
   const textSplitter = new RecursiveCharacterTextSplitter({ chunkSize: 1000 });
   const docs = await textSplitter.createDocuments([pdfText]);
   /* Create the vectorstore */
   const vectorStore = await HNSWLib.fromDocuments(docs, new OpenAIEmbeddings());
   /* Create the chain */
   const chain = ConversationalRetrievalQAChain.fromLLM(
      model,
      vectorStore.asRetriever(),
      {
         memory: new BufferMemory({
         memoryKey: 'chat_history', // Must be set to 'chat_history'
         }),
      }
   );

   /* Ask it a question */
   const res = await chain.call({ question: userPrompt });
   log(chalk.green(res.text));
};

rl.question('Enter your prompt: ', (prompt) => {
   const userPrompt = prompt;
   // Close the readline interface
   rl.close(); 
   // query the model
   readPdf(userPrompt);
});
