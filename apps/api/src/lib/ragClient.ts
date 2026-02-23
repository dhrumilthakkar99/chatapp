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
const RAG_BASE_URL = RAG_SERVICE_URL.replace(/\/+$/, "");
const RAG_TIMEOUT_MS = Number(process.env.RAG_TIMEOUT_MS || 45000);

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
  const queryUrl = `${RAG_BASE_URL}/query`;
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), RAG_TIMEOUT_MS);
  try {
    const response = await fetch(queryUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal: controller.signal,
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      const detail = (await response.text()).slice(0, 500);
      throw new Error(
        `RAG query failed: ${response.status} ${response.statusText} @ ${queryUrl}; body=${detail || "<empty>"}`
      );
    }

    const data = await response.json();
    return ragResponseSchema.parse(data);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    console.error("RAG upstream call failed", {
      queryUrl,
      timeoutMs: RAG_TIMEOUT_MS,
      error: message,
    });
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
  } finally {
    clearTimeout(timeout);
  }
}
