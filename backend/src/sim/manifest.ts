import { execFileSync } from "node:child_process";
import { writeFileSync } from "node:fs";
import { dirname, extname, join, basename } from "node:path";

export function manifestPathFor(outPath: string): string {
  const extension = extname(outPath);
  const stem = extension ? basename(outPath, extension) : basename(outPath);
  return join(dirname(outPath), `${stem}.manifest.json`);
}

export function writeManifestFor(outPath: string, manifest: Record<string, unknown>): string {
  const path = manifestPathFor(outPath);
  writeFileSync(path, JSON.stringify(withGitMetadata(manifest), null, 2) + "\n", "utf8");
  return path;
}

export function withGitMetadata(manifest: Record<string, unknown>): Record<string, unknown> {
  return { ...manifest, git: gitInfo() };
}

function gitInfo() {
  try {
    const sha = execFileSync("git", ["rev-parse", "HEAD"], { encoding: "utf8" }).trim();
    const status = execFileSync("git", ["status", "--short"], { encoding: "utf8" }).trim();
    return { sha, dirty: status.length > 0 };
  } catch {
    return { sha: null, dirty: null };
  }
}
