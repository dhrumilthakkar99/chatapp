import { create } from "zustand";
import { persist } from "zustand/middleware";
import type {
  ChatMessage,
  RetrievalChunk,
  SandboxCell,
  SandboxRunResult,
  TextSpan,
  WorkspaceTab,
} from "../lib/types";

const CODE_INTENT_RE = /\b(code|python|script|function|class|debug|refactor|notebook|analysis|plot|pandas|numpy|sql)\b/i;
const DOC_INTENT_RE = /\b(document|pdf|page|citation|snippet|source|table|figure|extract)\b/i;

export interface AppState {
  sessionId: string;
  input: string;
  selectedContext: string;
  messages: ChatMessage[];
  retrievalChunks: RetrievalChunk[];
  workspaceVisible: boolean;
  activeTab: WorkspaceTab;
  documentName: string;
  documentKind: "none" | "pdf" | "txt";
  documentText: string;
  documentUrl: string;
  documentPageCount: number;
  activePage: number;
  highlightTerms: string[];
  activeHighlightSpan: TextSpan | null;
  sandboxCode: string;
  sandboxCells: SandboxCell[];
  sandboxLast: SandboxRunResult;
  setInput: (text: string) => void;
  clearInput: () => void;
  addMessage: (message: ChatMessage) => void;
  setRetrievalChunks: (chunks: RetrievalChunk[]) => void;
  setWorkspaceVisible: (visible: boolean) => void;
  setActiveTab: (tab: WorkspaceTab) => void;
  setDocumentPdf: (name: string, url: string) => void;
  setDocumentTxt: (name: string, text: string) => void;
  setDocumentPageCount: (count: number) => void;
  setActivePage: (page: number) => void;
  setHighlightTerms: (terms: string[]) => void;
  setActiveHighlightSpan: (span: TextSpan | null) => void;
  appendSelectedContext: (text: string) => void;
  clearSelectedContext: () => void;
  setSandboxCode: (code: string) => void;
  appendSandboxCell: (cell: SandboxCell) => void;
  setSandboxLast: (result: SandboxRunResult) => void;
  resetSandbox: () => void;
  openWorkspaceByIntent: (text: string) => void;
}

function createSessionId(): string {
  return crypto.randomUUID ? crypto.randomUUID() : `s_${Date.now()}`;
}

export const useAppStore = create<AppState>()(
  persist(
    (set) => ({
      sessionId: createSessionId(),
      input: "",
      selectedContext: "",
      messages: [],
      retrievalChunks: [],
      workspaceVisible: false,
      activeTab: "document",
      documentName: "",
      documentKind: "none",
      documentText: "",
      documentUrl: "",
      documentPageCount: 1,
      activePage: 1,
      highlightTerms: [],
      activeHighlightSpan: null,
      sandboxCode: "",
      sandboxCells: [],
      sandboxLast: { stdout: "", stderr: "", images: [] },
      setInput: (text) => set({ input: text }),
      clearInput: () => set({ input: "" }),
      addMessage: (message) => set((state) => ({ messages: [...state.messages, message] })),
      setRetrievalChunks: (retrievalChunks) => set({ retrievalChunks }),
      setWorkspaceVisible: (workspaceVisible) => set({ workspaceVisible }),
      setActiveTab: (activeTab) => set({ activeTab }),
      setDocumentPdf: (documentName, documentUrl) =>
        set({
          documentName,
          documentUrl,
          documentKind: "pdf",
          workspaceVisible: true,
          activeTab: "document",
          activePage: 1,
          highlightTerms: [],
          activeHighlightSpan: null,
        }),
      setDocumentTxt: (documentName, documentText) =>
        set({
          documentName,
          documentText,
          documentKind: "txt",
          workspaceVisible: true,
          activeTab: "document",
          activePage: 1,
          highlightTerms: [],
          activeHighlightSpan: null,
        }),
      setDocumentPageCount: (documentPageCount) => set({ documentPageCount }),
      setActivePage: (activePage) => set({ activePage }),
      setHighlightTerms: (highlightTerms) => set({ highlightTerms }),
      setActiveHighlightSpan: (activeHighlightSpan) => set({ activeHighlightSpan }),
      appendSelectedContext: (text) =>
        set((state) => ({
          selectedContext: state.selectedContext ? `${state.selectedContext}\n\n${text}` : text,
        })),
      clearSelectedContext: () => set({ selectedContext: "" }),
      setSandboxCode: (sandboxCode) => set({ sandboxCode }),
      appendSandboxCell: (cell) => set((state) => ({ sandboxCells: [...state.sandboxCells, cell] })),
      setSandboxLast: (sandboxLast) => set({ sandboxLast }),
      resetSandbox: () =>
        set({
          sandboxCode: "",
          sandboxCells: [],
          sandboxLast: { stdout: "", stderr: "", images: [] },
        }),
      openWorkspaceByIntent: (text) => {
        if (CODE_INTENT_RE.test(text || "") || DOC_INTENT_RE.test(text || "")) {
          set({ workspaceVisible: true });
        }
      },
    }),
    {
      name: "chatqna-web-store",
      partialize: (state) => ({
        sessionId: state.sessionId,
        messages: state.messages,
        retrievalChunks: state.retrievalChunks,
        workspaceVisible: state.workspaceVisible,
        activeTab: state.activeTab,
        documentName: state.documentName,
        documentKind: state.documentKind,
        documentText: state.documentText,
        documentPageCount: state.documentPageCount,
        activePage: state.activePage,
        highlightTerms: state.highlightTerms,
        activeHighlightSpan: state.activeHighlightSpan,
        sandboxCode: state.sandboxCode,
        sandboxCells: state.sandboxCells,
        sandboxLast: state.sandboxLast,
      }),
    }
  )
);
