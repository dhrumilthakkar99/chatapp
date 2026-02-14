import type { ChatMessage, Citation, RetrievalChunk } from "../lib/types";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://localhost:8787";

export interface DocumentPayload {
  documentName?: string;
  documentKind?: "pdf" | "txt";
  documentText?: string;
}

export interface ChatRequest {
  sessionId: string;
  message: string;
  history: ChatMessage[];
  authToken?: string | null;
  document?: DocumentPayload;
}

export interface ChatResponse {
  answer: string;
  citations: Citation[];
  retrievedChunks: RetrievalChunk[];
}

export async function sendChat(request: ChatRequest): Promise<ChatResponse> {
  try {
    const response = await fetch(`${API_BASE}/api/chat`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...(request.authToken ? { Authorization: `Bearer ${request.authToken}` } : {}),
      },
      body: JSON.stringify({
        sessionId: request.sessionId,
        message: request.message,
        history: request.history,
        document: request.document,
      }),
    });

    if (!response.ok) {
      throw new Error(`Chat request failed: ${response.status}`);
    }

    return (await response.json()) as ChatResponse;
  } catch (error) {
    const fallback: RetrievalChunk[] = [
      {
        id: "S1",
        page: 1,
        chunkType: "main_text",
        text: "Backend not reachable. This is a local fallback response for UI iteration.",
      },
    ];

    return {
      answer:
        "I could not reach the API gateway. The UI is running in local fallback mode so you can keep iterating.",
      citations: [
        { id: "S1", page: 1, chunkType: "main_text", text: fallback[0].text },
      ],
      retrievedChunks: fallback,
    };
  }
}
