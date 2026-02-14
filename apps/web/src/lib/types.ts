export type Role = "user" | "assistant";

export interface TextSpan {
  startOffset: number;
  endOffset: number;
}

export interface Citation {
  id: string;
  page: number;
  chunkType: string;
  text: string;
  startOffset?: number;
  endOffset?: number;
  sourceDocument?: string;
}

export interface RetrievalChunk {
  id: string;
  page: number;
  chunkType: string;
  text: string;
  startOffset?: number;
  endOffset?: number;
  sourceDocument?: string;
  score?: number;
}

export interface ChatMessage {
  id: string;
  role: Role;
  content: string;
  citations?: Citation[];
  createdAt: number;
}

export type WorkspaceTab = "document" | "retrieval" | "sandbox" | "history";

export interface SandboxRunResult {
  stdout: string;
  stderr: string;
  images: string[];
}

export interface SandboxCell extends SandboxRunResult {
  id: string;
  code: string;
  createdAt: number;
}
