import { z } from "zod";

const ragChunkSchema = z.object({
  id: z.string(),
  page: z.number().int().default(1),
  chunkType: z.string().default("main_text"),
  text: z.string(),
  startOffset: z.number().int().optional(),
  endOffset: z.number().int().optional(),
  sourceDocument: z.string().optional(),
  score: z.number().optional(),
});

export const ragResponseSchema = z.object({
  answer: z.string(),
  retrievedChunks: z.array(ragChunkSchema).default([]),
  citations: z
    .array(
      z.object({
        id: z.string(),
        page: z.number().int().default(1),
        chunkType: z.string().default("main_text"),
        text: z.string(),
        startOffset: z.number().int().optional(),
        endOffset: z.number().int().optional(),
        sourceDocument: z.string().optional(),
      })
    )
    .default([]),
});

export type RagResponse = z.infer<typeof ragResponseSchema>;

const RAG_SERVICE_URL = process.env.RAG_SERVICE_URL || "http://localhost:8002";

export async function queryRag(payload: {
  sessionId: string;
  message: string;
  history: Array<{ role: string; content: string }>;
  topK?: number;
  userId: string;
  document?: {
    documentName?: string;
    documentKind?: "pdf" | "txt";
    documentText?: string;
  };
}): Promise<RagResponse> {
  try {
    const response = await fetch(`${RAG_SERVICE_URL}/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      throw new Error(`RAG query failed with status ${response.status}`);
    }

    const data = await response.json();
    return ragResponseSchema.parse(data);
  } catch (error) {
    return {
      answer:
        "RAG service is not reachable yet. This is an orchestrator fallback response so frontend integration can proceed.",
      retrievedChunks: [
        {
          id: "S1",
          page: 1,
          chunkType: "main_text",
          text: "RAG service unavailable; fallback chunk emitted by API orchestrator.",
          startOffset: 0,
          endOffset: 58,
          sourceDocument: payload.document?.documentName,
        },
      ],
      citations: [
        {
          id: "S1",
          page: 1,
          chunkType: "main_text",
          text: "RAG service unavailable; fallback chunk emitted by API orchestrator.",
          startOffset: 0,
          endOffset: 58,
          sourceDocument: payload.document?.documentName,
        },
      ],
    };
  }
}
