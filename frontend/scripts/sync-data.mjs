// Copies the generated benchmark showcase (../data/frontend/showcase.json, built by
// scripts/build_showcase.py and checked against docs/results_log.md) into src/data so the
// landing page reads its numbers from the same file the backend serves. Never edit the copy.
import { copyFileSync, existsSync, mkdirSync, readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const src = resolve(here, "../../data/frontend/showcase.json");
const dst = resolve(here, "../src/data/showcase.json");

if (!existsSync(src)) {
  if (existsSync(dst)) {
    console.warn(`[sync-data] ${src} not found; keeping the existing copy`);
    process.exit(0);
  }
  console.error(`[sync-data] ${src} not found. Build it with scripts/build_showcase.py`);
  process.exit(1);
}
const data = JSON.parse(readFileSync(src, "utf-8"));
if (data.schema_version !== 1) {
  console.error(`[sync-data] unexpected showcase schema_version ${data.schema_version}`);
  process.exit(1);
}
mkdirSync(dirname(dst), { recursive: true });
copyFileSync(src, dst);
console.log(`[sync-data] showcase.json (${data.cards.length} cards, ${data.checks_against_log} log checks) -> src/data`);
