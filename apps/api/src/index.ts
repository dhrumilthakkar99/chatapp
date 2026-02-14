import "dotenv/config";
import cors from "cors";
import express from "express";
import { clerkMiddleware } from "@clerk/express";
import { ensureSchema } from "./lib/db.js";
import { chatRouter } from "./routes/chat.js";
import { healthRouter } from "./routes/health.js";

const app = express();
const port = Number(process.env.PORT || 8787);

app.use(
  cors({
    origin: process.env.CORS_ORIGIN?.split(",") || ["http://localhost:5173"],
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
