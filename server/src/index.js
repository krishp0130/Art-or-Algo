import { randomUUID } from "crypto";
import cors from "cors";
import express from "express";
import fs from "fs/promises";
import fsSync from "fs";
import multer from "multer";
import path from "path";
import { fileURLToPath } from "url";
import { spawn } from "child_process";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.resolve(__dirname, "..", "..");
const uploadsRoot = path.join(__dirname, "..", "uploads");
const uploadsIn = path.join(uploadsRoot, "in");
const clientDist = path.join(projectRoot, "client", "dist");

const venvUnix = path.join(projectRoot, ".venv", "bin", "python");
const venvWin = path.join(projectRoot, ".venv", "Scripts", "python.exe");
const PYTHON_BIN =
  process.env.PYTHON_BIN ||
  (fsSync.existsSync(venvUnix)
    ? venvUnix
    : fsSync.existsSync(venvWin)
      ? venvWin
      : "python3");

const METRICS_PATH =
  process.env.METRICS_PATH ||
  path.join(projectRoot, "models", "metrics.json");
const FAILURES_PATH =
  process.env.FAILURES_PATH ||
  path.join(projectRoot, "server", "public", "failures", "failures.json");
const FAILURE_GALLERY_DIR =
  process.env.FAILURE_GALLERY_DIR ||
  path.join(projectRoot, "server", "public", "failures");

function resolveCheckpoint() {
  const finalP = path.join(projectRoot, "models", "final_vit.pth");
  const bestP = path.join(projectRoot, "models", "best_vit.pth");
  if (fsSync.existsSync(finalP)) return finalP;
  if (fsSync.existsSync(bestP)) return bestP;
  return finalP;
}

await fs.mkdir(uploadsIn, { recursive: true });
await fs.mkdir(FAILURE_GALLERY_DIR, { recursive: true });

const storage = multer.diskStorage({
  destination: (_req, _file, cb) => cb(null, uploadsIn),
  filename: (_req, file, cb) => {
    const ext = path.extname(file.originalname) || ".jpg";
    cb(null, `${randomUUID()}${ext}`);
  },
});

const upload = multer({
  storage,
  limits: { fileSize: 25 * 1024 * 1024 },
  fileFilter: (_req, file, cb) => {
    const ok = /^image\/(jpeg|png|webp|gif)$/i.test(file.mimetype);
    cb(ok ? null : new Error("Only image uploads are allowed"), ok);
  },
});

function runClassify(imagePath) {
  return new Promise((resolve, reject) => {
    const ckpt = resolveCheckpoint();
    const child = spawn(
      PYTHON_BIN,
      ["-m", "ml.inference", "--image", imagePath, "--checkpoint", ckpt],
      {
        cwd: projectRoot,
        stdio: ["ignore", "pipe", "pipe"],
        env: { ...process.env },
      }
    );
    let stdout = "";
    let stderr = "";
    child.stdout.on("data", (d) => {
      stdout += d.toString();
    });
    child.stderr.on("data", (d) => {
      stderr += d.toString();
    });
    child.on("error", reject);
    child.on("close", (code) => {
      if (code !== 0) {
        reject(
          new Error(stderr.trim() || `inference exited with code ${code}`)
        );
        return;
      }
      try {
        const line = stdout.trim().split("\n").filter(Boolean).pop();
        resolve(JSON.parse(line));
      } catch {
        reject(
          new Error(`Invalid JSON from inference: ${stdout.slice(0, 200)}`)
        );
      }
    });
  });
}

async function handleClassify(req, res) {
  if (!req.file) {
    res.status(400).json({ error: "Missing file field `image`" });
    return;
  }

  const imagePath = req.file.path;

  try {
    const result = await runClassify(imagePath);
    const b64 =
      result.attention_map_base64 ??
      result.heatmap_png_base64 ??
      null;
    res.json({
      prediction: result.prediction ?? result.label,
      confidence: result.confidence,
      heatmap_png_base64: b64,
      heatmap_data_url:
        result.heatmap_data_url ??
        (b64 ? `data:image/png;base64,${b64}` : null),
      label_id: result.label_id,
      probabilities: result.probabilities ?? null,
      rollout_mode: result.rollout_mode ?? null,
    });
  } catch (err) {
    console.error(err);
    res.status(500).json({
      error: "Inference failed",
      detail: err.message,
    });
  } finally {
    try {
      await fs.unlink(imagePath);
    } catch {
      /* ignore */
    }
  }
}

const app = express();
app.use(
  cors({
    origin: process.env.CORS_ORIGIN || true,
  })
);
app.use(express.json());

app.post("/api/classify", upload.single("image"), handleClassify);
app.post("/api/predict", upload.single("image"), handleClassify);

app.use("/api/failure-images", express.static(FAILURE_GALLERY_DIR));

function normalizeFailuresDoc(raw) {
  const parsed = typeof raw === "string" ? JSON.parse(raw) : raw;
  const arr = Array.isArray(parsed.failures) ? parsed.failures : [];
  return {
    failures: arr.map((f) => ({
      file: f.filename ?? f.file,
      trueLabel: f.true_label ?? f.trueLabel,
      predictedLabel: f.predicted_label ?? f.predictedLabel,
      confidence:
        f.confidence_in_predicted_class ?? f.confidence ?? null,
      hints: f.why_the_model_may_have_struggled ?? f.hints ?? [],
    })),
  };
}

app.get("/api/failures", async (_req, res) => {
  try {
    const raw = await fs.readFile(FAILURES_PATH, "utf8");
    const body = normalizeFailuresDoc(raw);
    res.json(body);
  } catch (e) {
    if (e.code === "ENOENT") {
      res.json({ failures: [] });
      return;
    }
    console.error(e);
    res.status(500).json({ error: "Could not read failures.json" });
  }
});

app.get("/api/stats", async (_req, res) => {
  try {
    const raw = await fs.readFile(METRICS_PATH, "utf8");
    res.type("application/json").send(raw);
  } catch (e) {
    if (e.code === "ENOENT") {
      res.status(404).json({
        error: "metrics.json not found",
        hint: "Set METRICS_PATH or run training to generate metrics.",
      });
      return;
    }
    console.error(e);
    res.status(500).json({ error: "Could not read metrics" });
  }
});

app.use((err, _req, res, next) => {
  if (!err) {
    next();
    return;
  }
  if (err instanceof multer.MulterError) {
    res.status(400).json({ error: err.message });
    return;
  }
  if (err.message === "Only image uploads are allowed") {
    res.status(400).json({ error: err.message });
    return;
  }
  next(err);
});

let clientReady = false;
try {
  await fs.access(path.join(clientDist, "index.html"));
  clientReady = true;
} catch {
  /* optional */
}

if (clientReady) {
  app.use(express.static(clientDist));
  app.get("*", (req, res) => {
    if (req.path.startsWith("/api")) {
      res.status(404).json({ error: "Not found" });
      return;
    }
    res.sendFile(path.join(clientDist, "index.html"));
  });
} else {
  console.warn(
    "client/dist missing — run `npm run build --prefix client` to enable static UI."
  );
}

const port = Number(process.env.PORT) || 3000;
app.listen(port, () => {
  console.log(`Server listening on http://localhost:${port}`);
});
