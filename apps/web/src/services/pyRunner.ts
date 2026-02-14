import type { SandboxRunResult } from "../lib/types";

type PyodideGlobal = {
  runPythonAsync: (code: string) => Promise<unknown>;
  runPython: (code: string) => unknown;
  setStdout: (opts: { batched: (msg: string) => void }) => void;
  setStderr: (opts: { batched: (msg: string) => void }) => void;
};

declare global {
  interface Window {
    loadPyodide?: (opts: { indexURL: string }) => Promise<PyodideGlobal>;
  }
}

const BLOCKED_PATTERNS = [
  /\bimport\s+os\b/,
  /\bimport\s+sys\b/,
  /\bimport\s+socket\b/,
  /\bimport\s+subprocess\b/,
  /\bimport\s+requests\b/,
  /\bimport\s+urllib\b/,
  /\bopen\s*\(/,
  /__import__\s*\(/,
];

let pyodide: PyodideGlobal | null = null;

async function ensureScript(): Promise<void> {
  if (window.loadPyodide) {
    return;
  }

  await new Promise<void>((resolve, reject) => {
    const script = document.createElement("script");
    script.src = "https://cdn.jsdelivr.net/pyodide/v0.26.4/full/pyodide.js";
    script.async = true;
    script.onload = () => resolve();
    script.onerror = () => reject(new Error("Failed to load Pyodide script"));
    document.head.appendChild(script);
  });
}

export async function initPyRunner(): Promise<void> {
  if (pyodide) {
    return;
  }

  await ensureScript();
  if (!window.loadPyodide) {
    throw new Error("Pyodide loader unavailable");
  }

  pyodide = await window.loadPyodide({
    indexURL: "https://cdn.jsdelivr.net/pyodide/v0.26.4/full/",
  });

  await pyodide.runPythonAsync(`
import matplotlib
matplotlib.use("AGG")
`);
}

export async function runPython(code: string): Promise<SandboxRunResult> {
  if (!pyodide) {
    await initPyRunner();
  }

  for (const pattern of BLOCKED_PATTERNS) {
    if (pattern.test(code)) {
      return {
        stdout: "",
        stderr: `Blocked by sandbox policy: ${pattern}`,
        images: [],
      };
    }
  }

  let stdout = "";
  let stderr = "";
  pyodide!.setStdout({ batched: (msg) => (stdout += `${msg}\n`) });
  pyodide!.setStderr({ batched: (msg) => (stderr += `${msg}\n`) });

  try {
    await pyodide!.runPythonAsync(`__chatqna_images__ = []`);
    await pyodide!.runPythonAsync(code);
    await pyodide!.runPythonAsync(`
try:
  import io, base64
  import matplotlib.pyplot as plt
  for fnum in plt.get_fignums():
    fig = plt.figure(fnum)
    _buf = io.BytesIO()
    fig.savefig(_buf, format="png", bbox_inches="tight")
    __chatqna_images__.append(base64.b64encode(_buf.getvalue()).decode("utf-8"))
  plt.close("all")
except Exception:
  pass
`);
    const images = (pyodide!.runPython("__chatqna_images__") as string[]) || [];

    return {
      stdout: stdout.trim(),
      stderr: stderr.trim(),
      images,
    };
  } catch (error) {
    return {
      stdout: stdout.trim(),
      stderr: `${stderr}\n${String(error)}`.trim(),
      images: [],
    };
  }
}

export async function resetPyRunner(): Promise<void> {
  pyodide = null;
}
