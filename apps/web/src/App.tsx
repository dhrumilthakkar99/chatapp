import { useEffect, useMemo, useRef, useState } from "react";
import { Bot, ChevronLeft, ChevronRight, FileText, FlaskConical, MessagesSquare, PanelRightClose, Search } from "lucide-react";
import clsx from "clsx";
import { SignIn, SignedIn, SignedOut, UserButton, useAuth, useUser } from "@clerk/clerk-react";
import type { Citation, ChatMessage, RetrievalChunk, TextSpan, WorkspaceTab } from "./lib/types";
import { sendChat } from "./services/api";
import { initPyRunner, resetPyRunner, runPython } from "./services/pyRunner";
import { useAppStore } from "./store/useAppStore";
import * as pdfjs from "pdfjs-dist";

(pdfjs as unknown as { GlobalWorkerOptions: { workerSrc: string } }).GlobalWorkerOptions.workerSrc =
  "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/4.10.38/pdf.worker.min.mjs";

function parseCitations(answer: string, retrievedChunks: RetrievalChunk[]): Citation[] {
  const ids = [...answer.matchAll(/\[(S\d+)\]/g)].map((m) => m[1]);
  const uniq = Array.from(new Set(ids));
  return uniq
    .map((id) => retrievedChunks.find((c) => c.id === id))
    .filter((c): c is RetrievalChunk => Boolean(c))
    .map((c) => ({ id: c.id, page: c.page, chunkType: c.chunkType, text: c.text }));
}

function renderWithCitationButtons(content: string, citations: Citation[] | undefined): { text: string; cites: Citation[] } {
  return { text: content, cites: citations || [] };
}

function normalizeSpan(span: TextSpan | null, textLength: number): TextSpan | null {
  if (!span || textLength <= 0) return null;
  const start = Math.max(0, Math.min(textLength - 1, Math.floor(span.startOffset)));
  const end = Math.max(start + 1, Math.min(textLength, Math.floor(span.endOffset)));
  if (Number.isNaN(start) || Number.isNaN(end)) return null;
  return { startOffset: start, endOffset: end };
}

function renderTextWithSpan(text: string, activeSpan: TextSpan | null): JSX.Element {
  const safeText = text || "";
  const span = normalizeSpan(activeSpan, safeText.length);
  if (!span) {
    return <>{safeText}</>;
  }

  const context = 1600;
  const windowStart = Math.max(0, span.startOffset - context);
  const windowEnd = Math.min(safeText.length, span.endOffset + context);
  const before = safeText.slice(windowStart, span.startOffset);
  const marked = safeText.slice(span.startOffset, span.endOffset);
  const after = safeText.slice(span.endOffset, windowEnd);
  const hasPrefix = windowStart > 0;
  const hasSuffix = windowEnd < safeText.length;

  return (
    <>
      {hasPrefix ? "...\n" : ""}
      {before}
      <mark className="rounded bg-amber-300 px-1 text-slate-950">{marked}</mark>
      {after}
      {hasSuffix ? "\n..." : ""}
    </>
  );
}

function useWorkspaceTabs() {
  const activeTab = useAppStore((s) => s.activeTab);
  const setActiveTab = useAppStore((s) => s.setActiveTab);
  const tabs: Array<{ key: WorkspaceTab; label: string; icon: JSX.Element }> = [
    { key: "document", label: "Document", icon: <FileText size={14} /> },
    { key: "retrieval", label: "Retrieval", icon: <Search size={14} /> },
    { key: "sandbox", label: "Sandbox", icon: <FlaskConical size={14} /> },
    { key: "history", label: "History", icon: <MessagesSquare size={14} /> },
  ];
  return { activeTab, setActiveTab, tabs };
}

function DocumentTab() {
  const documentKind = useAppStore((s) => s.documentKind);
  const documentName = useAppStore((s) => s.documentName);
  const documentText = useAppStore((s) => s.documentText);
  const documentUrl = useAppStore((s) => s.documentUrl);
  const documentPageCount = useAppStore((s) => s.documentPageCount);
  const activePage = useAppStore((s) => s.activePage);
  const highlightTerms = useAppStore((s) => s.highlightTerms);
  const activeHighlightSpan = useAppStore((s) => s.activeHighlightSpan);
  const setDocumentPdf = useAppStore((s) => s.setDocumentPdf);
  const setDocumentTxt = useAppStore((s) => s.setDocumentTxt);
  const setDocumentPageCount = useAppStore((s) => s.setDocumentPageCount);
  const setActivePage = useAppStore((s) => s.setActivePage);
  const setActiveHighlightSpan = useAppStore((s) => s.setActiveHighlightSpan);
  const appendSelectedContext = useAppStore((s) => s.appendSelectedContext);
  const [uploading, setUploading] = useState(false);

  const onUpload: React.ChangeEventHandler<HTMLInputElement> = async (event) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setUploading(true);
    try {
      if (file.type === "application/pdf") {
        const buf = await file.arrayBuffer();
        const blob = new Blob([buf], { type: "application/pdf" });
        const objectUrl = URL.createObjectURL(blob);
        setDocumentPdf(file.name, objectUrl);
        setActiveHighlightSpan(null);

        const loadingTask = pdfjs.getDocument({ data: new Uint8Array(buf) });
        const pdf = await loadingTask.promise;
        setDocumentPageCount(pdf.numPages || 1);
      } else {
        const text = await file.text();
        setDocumentTxt(file.name, text);
        setDocumentPageCount(1);
        setActiveHighlightSpan(null);
      }
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="space-y-3">
      <label className="block">
        <span className="mb-1 block text-xs text-slate-300">Upload PDF / TXT</span>
        <input type="file" accept=".pdf,.txt" onChange={onUpload} className="block w-full text-xs" />
      </label>

      {uploading && <p className="text-xs text-slate-400">Processing document...</p>}

      {documentKind === "none" && <p className="text-sm text-slate-400">No active document yet.</p>}

      {documentKind !== "none" && (
        <>
          <div className="rounded-lg border border-border bg-panel2 p-2 text-xs">
            <p className="font-medium text-slate-200">{documentName}</p>
            <p className="text-slate-400">Highlights: {highlightTerms.join(", ") || "none"}</p>
          </div>

          {documentKind === "pdf" ? (
            <>
              <div className="flex items-center justify-between rounded-lg border border-border bg-panel px-2 py-1 text-xs">
                <button
                  className="rounded bg-slate-800 px-2 py-1 disabled:opacity-50"
                  onClick={() => setActivePage(Math.max(1, activePage - 1))}
                  disabled={activePage <= 1}
                >
                  <ChevronLeft size={14} />
                </button>
                <span>
                  Page {activePage} / {documentPageCount}
                </span>
                <button
                  className="rounded bg-slate-800 px-2 py-1 disabled:opacity-50"
                  onClick={() => setActivePage(Math.min(documentPageCount, activePage + 1))}
                  disabled={activePage >= documentPageCount}
                >
                  <ChevronRight size={14} />
                </button>
              </div>
              <iframe
                title="document-viewer"
                src={`${documentUrl}#page=${activePage}`}
                className="h-[420px] w-full rounded-lg border border-border bg-black"
              />
            </>
          ) : (
            <div className="code-font h-[420px] overflow-auto rounded-lg border border-border bg-panel p-3 text-xs whitespace-pre-wrap">
              {renderTextWithSpan(documentText, activeHighlightSpan)}
            </div>
          )}

          <button
            className="rounded-md border border-border bg-panel px-3 py-1 text-xs hover:bg-panel2"
            onClick={() => {
              if (activeHighlightSpan && documentKind === "txt") {
                const start = Math.max(0, activeHighlightSpan.startOffset);
                const end = Math.min(documentText.length, activeHighlightSpan.endOffset);
                appendSelectedContext(documentText.slice(start, end));
                return;
              }
              appendSelectedContext(`Document focus: ${documentName}, page ${activePage}`);
            }}
          >
            Send highlighted context to chat
          </button>
        </>
      )}
    </div>
  );
}

function RetrievalTab() {
  const retrievalChunks = useAppStore((s) => s.retrievalChunks);
  const setActivePage = useAppStore((s) => s.setActivePage);
  const setHighlightTerms = useAppStore((s) => s.setHighlightTerms);
  const setActiveHighlightSpan = useAppStore((s) => s.setActiveHighlightSpan);
  const appendSelectedContext = useAppStore((s) => s.appendSelectedContext);
  const setActiveTab = useAppStore((s) => s.setActiveTab);

  return (
    <div className="space-y-2">
      {retrievalChunks.length === 0 && <p className="text-sm text-slate-400">No retrieved chunks yet.</p>}
      {retrievalChunks.map((chunk) => (
        <div key={chunk.id} className="rounded-lg border border-border bg-panel p-2 text-xs">
          <div className="mb-1 flex items-center justify-between text-slate-300">
            <span>{chunk.id}</span>
            <span>
              p.{chunk.page} | {chunk.chunkType}
            </span>
          </div>
          <p className="mb-2 line-clamp-4 text-slate-200">{chunk.text}</p>
          <div className="flex flex-wrap gap-2">
            <button
              className="rounded border border-border px-2 py-1 hover:bg-panel2"
              onClick={() => {
                setActivePage(chunk.page || 1);
                if (
                  Number.isFinite(chunk.startOffset) &&
                  Number.isFinite(chunk.endOffset) &&
                  typeof chunk.startOffset === "number" &&
                  typeof chunk.endOffset === "number"
                ) {
                  setActiveHighlightSpan({ startOffset: chunk.startOffset, endOffset: chunk.endOffset });
                  setHighlightTerms([]);
                } else {
                  setActiveHighlightSpan(null);
                  setHighlightTerms((chunk.text || "").split(/\s+/).slice(0, 6));
                }
                setActiveTab("document");
              }}
            >
              Open in viewer
            </button>
            <button
              className="rounded border border-border px-2 py-1 hover:bg-panel2"
              onClick={() => appendSelectedContext(`[${chunk.id}] ${chunk.text}`)}
            >
              Add to chat context
            </button>
          </div>
        </div>
      ))}
    </div>
  );
}

function SandboxTab() {
  const sandboxCode = useAppStore((s) => s.sandboxCode);
  const sandboxLast = useAppStore((s) => s.sandboxLast);
  const sandboxCells = useAppStore((s) => s.sandboxCells);
  const messages = useAppStore((s) => s.messages);
  const setSandboxCode = useAppStore((s) => s.setSandboxCode);
  const setSandboxLast = useAppStore((s) => s.setSandboxLast);
  const appendSandboxCell = useAppStore((s) => s.appendSandboxCell);
  const resetSandboxState = useAppStore((s) => s.resetSandbox);
  const [running, setRunning] = useState(false);

  const latestAssistantCode = useMemo(() => {
    const lastAssistant = [...messages].reverse().find((m) => m.role === "assistant");
    if (!lastAssistant) return "";
    const match = /```python\s*([\s\S]*?)```/i.exec(lastAssistant.content);
    return match?.[1]?.trim() || "";
  }, [messages]);

  const onRun = async () => {
    setRunning(true);
    try {
      await initPyRunner();
      const result = await runPython(sandboxCode);
      setSandboxLast(result);
      appendSandboxCell({
        id: crypto.randomUUID(),
        code: sandboxCode,
        ...result,
        createdAt: Date.now(),
      });
    } finally {
      setRunning(false);
    }
  };

  const onReset = async () => {
    await resetPyRunner();
    resetSandboxState();
  };

  const exportNotebook = () => {
    const notebook = {
      nbformat: 4,
      nbformat_minor: 5,
      metadata: { language_info: { name: "python" } },
      cells: sandboxCells.map((cell) => ({
        cell_type: "code",
        execution_count: null,
        metadata: {},
        source: cell.code,
        outputs: [
          ...(cell.stdout ? [{ output_type: "stream", name: "stdout", text: cell.stdout }] : []),
          ...(cell.stderr ? [{ output_type: "stream", name: "stderr", text: cell.stderr }] : []),
        ],
      })),
    };

    const blob = new Blob([JSON.stringify(notebook, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "workspace.ipynb";
    a.click();
    URL.revokeObjectURL(url);
  };

  const exportMarkdown = () => {
    const lines: string[] = ["# Workspace Report", ""];
    sandboxCells.forEach((cell, index) => {
      lines.push(`## Cell ${index + 1}`);
      lines.push("```python");
      lines.push(cell.code);
      lines.push("```");
      if (cell.stdout) {
        lines.push("\nstdout:\n```text");
        lines.push(cell.stdout);
        lines.push("```");
      }
      if (cell.stderr) {
        lines.push("\nstderr:\n```text");
        lines.push(cell.stderr);
        lines.push("```");
      }
      lines.push("");
    });

    const blob = new Blob([lines.join("\n")], { type: "text/markdown" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "workspace.md";
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-2">
      <div className="flex flex-wrap gap-2">
        <button
          className="rounded border border-border px-2 py-1 text-xs hover:bg-panel2"
          onClick={() => latestAssistantCode && setSandboxCode(latestAssistantCode)}
        >
          Use last assistant code
        </button>
        <button className="rounded border border-border px-2 py-1 text-xs hover:bg-panel2" onClick={onReset}>
          Reset kernel + clear
        </button>
      </div>

      <textarea
        className="code-font h-44 w-full rounded-lg border border-border bg-panel p-2 text-xs"
        value={sandboxCode}
        onChange={(e) => setSandboxCode(e.target.value)}
        placeholder="Write Python here..."
      />

      <button
        className="w-full rounded bg-accent px-3 py-2 text-sm font-semibold text-black disabled:opacity-60"
        onClick={onRun}
        disabled={running}
      >
        {running ? "Running..." : "Run"}
      </button>

      {(sandboxLast.stdout || sandboxLast.stderr || sandboxLast.images.length > 0) && (
        <div className="space-y-2 rounded-lg border border-border bg-panel p-2 text-xs">
          {sandboxLast.stdout && (
            <>
              <p className="text-slate-300">stdout</p>
              <pre className="code-font whitespace-pre-wrap">{sandboxLast.stdout}</pre>
            </>
          )}
          {sandboxLast.stderr && (
            <>
              <p className="text-rose-300">stderr</p>
              <pre className="code-font whitespace-pre-wrap text-rose-200">{sandboxLast.stderr}</pre>
            </>
          )}
          {sandboxLast.images.length > 0 && (
            <div className="space-y-2">
              {sandboxLast.images.map((img, idx) => (
                <img
                  key={`${idx}_${img.slice(0, 12)}`}
                  src={`data:image/png;base64,${img}`}
                  alt={`plot-${idx}`}
                  className="rounded border border-border"
                />
              ))}
            </div>
          )}
        </div>
      )}

      <div className="flex gap-2">
        <button className="w-full rounded border border-border px-2 py-1 text-xs" onClick={exportNotebook}>
          Export .ipynb
        </button>
        <button className="w-full rounded border border-border px-2 py-1 text-xs" onClick={exportMarkdown}>
          Export .md
        </button>
      </div>
    </div>
  );
}

function HistoryTab() {
  const messages = useAppStore((s) => s.messages);
  const appendSelectedContext = useAppStore((s) => s.appendSelectedContext);
  const setSandboxCode = useAppStore((s) => s.setSandboxCode);

  return (
    <div className="space-y-2">
      {messages.slice(-14).map((message) => (
        <div key={message.id} className="rounded-lg border border-border bg-panel p-2 text-xs">
          <p className="mb-1 font-medium text-slate-300">{message.role === "user" ? "You" : "Assistant"}</p>
          <p className="line-clamp-4">{message.content}</p>
          <div className="mt-2 flex gap-2">
            <button
              className="rounded border border-border px-2 py-1 hover:bg-panel2"
              onClick={() => appendSelectedContext(message.content)}
            >
              Add to chat
            </button>
            <button
              className="rounded border border-border px-2 py-1 hover:bg-panel2"
              onClick={() => setSandboxCode(message.content)}
            >
              Send to sandbox
            </button>
          </div>
        </div>
      ))}
    </div>
  );
}

function WorkspacePanel({ onClose }: { onClose: () => void }) {
  const { tabs, activeTab, setActiveTab } = useWorkspaceTabs();

  return (
    <aside className="sticky top-20 flex h-[calc(100vh-136px)] flex-col rounded-xl border border-border/80 bg-panel/90 p-3 shadow-panel backdrop-blur">
      <div className="mb-3 flex items-center justify-between">
        <p className="text-sm font-semibold tracking-wide text-slate-200">Workspace</p>
        <button
          className="rounded-md border border-border bg-panel2 p-1.5 text-slate-300 hover:bg-panel hover:text-white"
          onClick={onClose}
          title="Close workspace"
          aria-label="Close workspace"
        >
          <PanelRightClose size={14} />
        </button>
      </div>

      <div className="mb-3 flex flex-wrap gap-2">
        {tabs.map((tab) => (
          <button
            key={tab.key}
            onClick={() => setActiveTab(tab.key)}
            className={clsx(
              "flex items-center gap-1 rounded-md border px-2 py-1 text-xs transition-colors",
              activeTab === tab.key
                ? "border-accent bg-accent/20 text-slate-100"
                : "border-border bg-panel2 text-slate-300 hover:bg-panel"
            )}
          >
            {tab.icon}
            <span>{tab.label}</span>
          </button>
        ))}
      </div>

      <div className="flex-1 overflow-y-auto pr-1">
        {activeTab === "document" && <DocumentTab />}
        {activeTab === "retrieval" && <RetrievalTab />}
        {activeTab === "sandbox" && <SandboxTab />}
        {activeTab === "history" && <HistoryTab />}
      </div>
    </aside>
  );
}

export default function App() {
  const { isLoaded, isSignedIn, getToken } = useAuth();
  const { user } = useUser();
  const messages = useAppStore((s) => s.messages);
  const input = useAppStore((s) => s.input);
  const selectedContext = useAppStore((s) => s.selectedContext);
  const workspaceVisible = useAppStore((s) => s.workspaceVisible);
  const sessionId = useAppStore((s) => s.sessionId);
  const documentName = useAppStore((s) => s.documentName);
  const documentKind = useAppStore((s) => s.documentKind);
  const documentText = useAppStore((s) => s.documentText);
  const setInput = useAppStore((s) => s.setInput);
  const clearInput = useAppStore((s) => s.clearInput);
  const addMessage = useAppStore((s) => s.addMessage);
  const clearSelectedContext = useAppStore((s) => s.clearSelectedContext);
  const setWorkspaceVisible = useAppStore((s) => s.setWorkspaceVisible);
  const openWorkspaceByIntent = useAppStore((s) => s.openWorkspaceByIntent);
  const setRetrievalChunks = useAppStore((s) => s.setRetrievalChunks);
  const setHighlightTerms = useAppStore((s) => s.setHighlightTerms);
  const setActiveHighlightSpan = useAppStore((s) => s.setActiveHighlightSpan);
  const setActivePage = useAppStore((s) => s.setActivePage);
  const [isSending, setIsSending] = useState(false);
  const chatViewportRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const el = chatViewportRef.current;
    if (!el) return;
    el.scrollTo({ top: el.scrollHeight, behavior: messages.length > 1 ? "smooth" : "auto" });
  }, [messages.length]);

  const handleSubmit = async () => {
    if (!isSignedIn) return;
    const raw = input.trim();
    if (!raw || isSending) return;

    const prompt = selectedContext ? `${selectedContext}\n\n${raw}` : raw;
    openWorkspaceByIntent(prompt);

    const userMessage: ChatMessage = {
      id: crypto.randomUUID(),
      role: "user",
      content: raw,
      createdAt: Date.now(),
    };

    addMessage(userMessage);
    clearInput();
    clearSelectedContext();
    setIsSending(true);

    try {
      const authToken = await getToken();
      if (!authToken) {
        throw new Error("No Clerk token available. Please sign in again.");
      }

      const history = [...messages, userMessage];
      const response = await sendChat({
        sessionId,
        message: prompt,
        history,
        authToken,
        document:
          documentKind === "none"
            ? undefined
            : {
                documentName: documentName || undefined,
                documentKind,
                documentText: documentKind === "txt" ? documentText : undefined,
              },
      });

      const citations = response.citations?.length
        ? response.citations
        : parseCitations(response.answer, response.retrievedChunks || []);

      const assistantMessage: ChatMessage = {
        id: crypto.randomUUID(),
        role: "assistant",
        content: response.answer,
        citations,
        createdAt: Date.now(),
      };

      addMessage(assistantMessage);
      setRetrievalChunks(response.retrievedChunks || []);
    } catch (error) {
      const message = error instanceof Error ? error.message : "Failed to send message.";
      addMessage({
        id: crypto.randomUUID(),
        role: "assistant",
        content: `Request failed: ${message}`,
        createdAt: Date.now(),
      });
    } finally {
      setIsSending(false);
    }
  };

  const handleCitationClick = (citation: Citation) => {
    setWorkspaceVisible(true);
    setActivePage(citation.page || 1);
    if (
      Number.isFinite(citation.startOffset) &&
      Number.isFinite(citation.endOffset) &&
      typeof citation.startOffset === "number" &&
      typeof citation.endOffset === "number"
    ) {
      setActiveHighlightSpan({ startOffset: citation.startOffset, endOffset: citation.endOffset });
      setHighlightTerms([]);
    } else {
      setActiveHighlightSpan(null);
      setHighlightTerms((citation.text || "").split(/\s+/).slice(0, 6));
    }
  };

  if (!isLoaded) {
    return (
      <main className="grid min-h-screen place-items-center text-slate-300">
        <p className="text-sm">Loading authentication...</p>
      </main>
    );
  }

  return (
    <div className="app-bg min-h-screen text-slate-100">
      <SignedOut>
        <main className="mx-auto flex min-h-screen max-w-[440px] items-center px-4">
          <div className="w-full rounded-xl border border-border bg-panel p-4 shadow-panel">
            <div className="mb-4 flex items-center gap-2">
              <Bot size={18} className="text-accent" />
              <h1 className="text-lg font-semibold">ChatQnA</h1>
            </div>
            <p className="mb-4 text-sm text-slate-300">
              Sign in with magic link or OAuth to use chat, document-grounded citations, and the analysis workspace.
            </p>
            <SignIn routing="hash" />
          </div>
        </main>
      </SignedOut>

      <SignedIn>
        <header className="sticky top-0 z-20 border-b border-border/70 bg-bg/80 backdrop-blur">
          <div className="mx-auto flex max-w-[1400px] items-center justify-between px-4 py-3">
            <div className="flex items-center gap-2">
              <Bot size={18} className="text-accent" />
              <h1 className="text-lg font-semibold">ChatQnA AI Lab</h1>
            </div>
            <div className="flex items-center gap-3 text-xs text-slate-400">
              <span>{user?.primaryEmailAddress?.emailAddress || user?.username || "Signed in"}</span>
              <UserButton />
            </div>
          </div>
        </header>

        <main className="mx-auto max-w-[1400px] px-4 pb-40 pt-4">
          <div className={workspaceVisible ? "grid gap-4 lg:grid-cols-[minmax(0,1fr)_420px]" : "mx-auto max-w-4xl"}>
            <section className="chat-panel rounded-xl border border-border/80 bg-panel/70 p-3 shadow-panel">
              <div
                ref={chatViewportRef}
                className={clsx(
                  "chat-scroll space-y-3 pr-1",
                  workspaceVisible ? "h-[calc(100vh-272px)]" : "h-[calc(100vh-258px)]"
                )}
              >
                {messages.length === 0 && (
                  <div className="rounded-lg border border-dashed border-border bg-panel2/60 p-8 text-center">
                    <p className="text-lg font-medium">Ask anything about docs or analysis</p>
                    <p className="mt-1 text-sm text-slate-400">
                      Workspace remains hidden for casual chat and appears contextually when needed.
                    </p>
                  </div>
                )}

                {messages.map((message) => {
                  const rendered = renderWithCitationButtons(message.content, message.citations);
                  return (
                    <article
                      key={message.id}
                      className={clsx(
                        "message-card rounded-lg border p-3 text-sm",
                        message.role === "user"
                          ? "ml-12 border-cyan-300/30 bg-cyan-400/10"
                          : "mr-12 border-border bg-panel2/80"
                      )}
                    >
                      <p className="mb-1 text-xs uppercase tracking-wide text-slate-400">
                        {message.role === "user" ? "You" : "Assistant"}
                      </p>
                      <p className="whitespace-pre-wrap leading-relaxed">{rendered.text}</p>
                      {rendered.cites.length > 0 && (
                        <div className="mt-2 flex flex-wrap gap-2">
                          {rendered.cites.map((citation) => (
                            <button
                              key={`${message.id}_${citation.id}`}
                              className="rounded border border-border bg-panel px-2 py-1 text-xs hover:bg-panel2"
                              onClick={() => handleCitationClick(citation)}
                            >
                              {citation.id} p.{citation.page}
                            </button>
                          ))}
                        </div>
                      )}
                    </article>
                  );
                })}
              </div>
            </section>

            {workspaceVisible && <WorkspacePanel onClose={() => setWorkspaceVisible(false)} />}
          </div>
        </main>

        {!workspaceVisible && (
          <button
            className="fixed right-4 top-1/2 z-40 -translate-y-1/2 rounded-l-lg border border-border/90 bg-panel2/95 px-2 py-3 text-xs text-slate-200 shadow-panel hover:bg-panel"
            onClick={() => setWorkspaceVisible(true)}
          >
            Open Lab
          </button>
        )}

        <footer className="fixed inset-x-0 bottom-0 z-50 border-t border-border/80 bg-gradient-to-t from-bg via-bg/95 to-bg/80 px-3 py-3 backdrop-blur">
          <div className={workspaceVisible ? "mx-auto max-w-[1400px]" : "mx-auto max-w-4xl"}>
            {selectedContext && (
              <div className="mb-2 rounded-md border border-border bg-panel2 p-2 text-xs">
                <div className="mb-1 flex items-center justify-between">
                  <span className="text-slate-300">Queued context for next message</span>
                  <button className="text-slate-400 hover:text-white" onClick={clearSelectedContext}>
                    clear
                  </button>
                </div>
                <p className="line-clamp-2 text-slate-200">{selectedContext}</p>
              </div>
            )}

            <div className="flex items-end gap-2">
              <textarea
                className="h-20 flex-1 resize-none rounded-xl border border-border/90 bg-panel px-3 py-2 text-sm outline-none focus:border-accent/70"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Message ChatQnA"
              />
              <button
                className="h-20 rounded-xl bg-accent px-5 text-sm font-semibold text-black disabled:opacity-60"
                disabled={!input.trim() || isSending}
                onClick={handleSubmit}
              >
                {isSending ? "..." : "Send"}
              </button>
            </div>
          </div>
        </footer>
      </SignedIn>
    </div>
  );
}
