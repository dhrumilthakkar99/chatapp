import { Pool } from "pg";

const DATABASE_URL = process.env.DATABASE_URL || "postgres://chatqna:chatqna@localhost:5432/chatqna";

export const pool = new Pool({ connectionString: DATABASE_URL });

export async function ensureSchema(): Promise<void> {
  await pool.query(`
    CREATE TABLE IF NOT EXISTS chat_messages (
      id BIGSERIAL PRIMARY KEY,
      session_id TEXT NOT NULL,
      user_id TEXT NOT NULL,
      role TEXT NOT NULL,
      content TEXT NOT NULL,
      metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
      created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );

    CREATE TABLE IF NOT EXISTS workspace_cells (
      id BIGSERIAL PRIMARY KEY,
      session_id TEXT NOT NULL,
      user_id TEXT NOT NULL,
      code TEXT NOT NULL,
      stdout TEXT NOT NULL DEFAULT '',
      stderr TEXT NOT NULL DEFAULT '',
      images_json JSONB NOT NULL DEFAULT '[]'::jsonb,
      created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );

    CREATE INDEX IF NOT EXISTS idx_chat_session ON chat_messages(session_id, created_at);
    CREATE INDEX IF NOT EXISTS idx_cells_session ON workspace_cells(session_id, created_at);
  `);
}

export async function saveMessage(params: {
  sessionId: string;
  userId: string;
  role: "user" | "assistant";
  content: string;
  metadata?: Record<string, unknown>;
}): Promise<void> {
  await pool.query(
    `
      INSERT INTO chat_messages(session_id, user_id, role, content, metadata)
      VALUES ($1, $2, $3, $4, $5::jsonb)
    `,
    [params.sessionId, params.userId, params.role, params.content, JSON.stringify(params.metadata || {})]
  );
}

export async function loadMessages(sessionId: string): Promise<Array<{ role: string; content: string }>> {
  const result = await pool.query(
    `
      SELECT role, content
      FROM chat_messages
      WHERE session_id = $1
      ORDER BY created_at ASC
      LIMIT 300
    `,
    [sessionId]
  );

  return result.rows.map((row: { role: string; content: string }) => ({
    role: String(row.role),
    content: String(row.content),
  }));
}
