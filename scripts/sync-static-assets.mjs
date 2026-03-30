#!/usr/bin/env node
/**
 * Copy metrics + failure gallery into client/public so the Vue build can load them
 * without the Express API (dashboard + gallery fallback; classify still needs the API or VITE_API_BASE).
 */
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(__dirname, "..");
const metricsSrc = path.join(root, "models", "metrics.json");
const failuresSrcDir = path.join(root, "server", "public", "failures");
const pub = path.join(root, "client", "public");
const pubFail = path.join(pub, "failures");

fs.mkdirSync(pubFail, { recursive: true });

if (fs.existsSync(metricsSrc)) {
  fs.copyFileSync(metricsSrc, path.join(pub, "metrics.json"));
  console.log("sync: models/metrics.json -> client/public/metrics.json");
} else {
  console.warn("sync: skip metrics (models/metrics.json missing)");
}

if (fs.existsSync(path.join(failuresSrcDir, "failures.json"))) {
  for (const name of fs.readdirSync(failuresSrcDir)) {
    const from = path.join(failuresSrcDir, name);
    if (fs.statSync(from).isFile() && name !== ".gitkeep") {
      fs.copyFileSync(from, path.join(pubFail, name));
    }
  }
  console.log("sync: server/public/failures/* -> client/public/failures/");
} else {
  console.warn("sync: skip failures (no failures.json yet — run ml/analyze_failures.py)");
}
