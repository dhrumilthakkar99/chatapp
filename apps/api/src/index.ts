import "dotenv/config";
import cors from "cors";
import express from "express";
import { clerkMiddleware } from "@clerk/express";
import { ensureSchema } from "./lib/db.js";
import { chatRouter } from "./routes/chat.js";
import { healthRouter } from "./routes/health.js";

const app = express();
const port = Number(process.env.PORT || 8787);

function normalizeOrigin(value: string): string | null {
  const raw = (value || "").trim();
  if (!raw) return null;
  try {
    const url = new URL(raw);
    return `${url.protocol}//${url.host}`;
  } catch {
    return null;
  }
}

const allowedOrigins = (process.env.CORS_ORIGIN || "http://localhost:5173")
  .split(",")
  .map((v) => normalizeOrigin(v))
  .filter((v): v is string => Boolean(v));

app.use(
  cors({
    origin: (origin, callback) => {
      if (!origin) {
        callback(null, true);
        return;
      }
      const normalized = normalizeOrigin(origin);
      if (normalized && allowedOrigins.includes(normalized)) {
        callback(null, true);
        return;
      }
      callback(new Error(`CORS origin not allowed: ${origin}`));
    },
    credentials: true,
  })
);
app.use(express.json({ limit: "4mb" }));
app.use(clerkMiddleware());

app.use(healthRouter);
app.use("/api", chatRouter);

async function boot(): Promise<void> {
  await ensureSchema();
  app.listen(port, () => {
    console.log(`chatqna-api listening on :${port}`);
  });
}

boot().catch((error) => {
  console.error("Failed to boot API", error);
  process.exit(1);
});
