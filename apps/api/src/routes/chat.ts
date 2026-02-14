import { Router } from "express";
import { getAuth, requireAuth } from "@clerk/express";
import { z } from "zod";
import { loadMessages, saveMessage } from "../lib/db.js";
import { queryRag } from "../lib/ragClient.js";

const chatRequestSchema = z.object({
  sessionId: z.string().min(1),
  message: z.string().min(1),
  history: z
    .array(
      z.object({
        role: z.enum(["user", "assistant"]),
        content: z.string(),
      })
    )
    .default([]),
  document: z
    .object({
      documentName: z.string().min(1).optional(),
      documentKind: z.enum(["pdf", "txt"]).optional(),
      documentText: z.string().optional(),
    })
    .optional(),
});

export const chatRouter = Router();

chatRouter.post("/chat", requireAuth(), async (req, res) => {
  try {
    const parsed = chatRequestSchema.parse(req.body);
    const auth = getAuth(req);
    if (!auth.userId) {
      res.status(401).json({ error: "Unauthorized: missing verified user identity" });
      return;
    }

    const userId = auth.userId;
    await saveMessage({
      sessionId: parsed.sessionId,
      userId,
      role: "user",
      content: parsed.message,
    });

    const history = parsed.history.length > 0 ? parsed.history : await loadMessages(parsed.sessionId);

    const rag = await queryRag({
      sessionId: parsed.sessionId,
      message: parsed.message,
      history,
      topK: 6,
      userId,
      document: parsed.document
        ? {
            documentName: parsed.document.documentName,
            documentKind: parsed.document.documentKind,
            documentText: parsed.document.documentText?.slice(0, 400_000),
          }
        : undefined,
    });

    await saveMessage({
      sessionId: parsed.sessionId,
      userId,
      role: "assistant",
      content: rag.answer,
      metadata: {
        citations: rag.citations,
        retrievedChunks: rag.retrievedChunks,
      },
    });

    res.json(rag);
  } catch (error) {
    res.status(400).json({
      error: "Invalid chat request",
      detail: String(error),
    });
  }
});
