// Throughput #4 (docs/ai-research/scoping/r12-selfplay-gate-throughput.md):
// persistent keep-alive transport for the serve_onnx /predict path.
//
// MCTS issues 200k+ /predict calls per training run. The previous code used
// bare global `fetch()` per call; without an explicit keep-alive Agent each
// call could pay a fresh TCP handshake to the local serve_onnx server.
//
// This helper is PURE TRANSPORT REUSE. It MUST send a byte-identical request
// body and return a response object parsed identically to the prior
// `fetch(...).json()` path. It does NOT change request ordering, payloads,
// retries, or error semantics. The only observable difference vs. bare
// `fetch()` is socket reuse, which cannot affect self-play / gate
// trajectories (proven by the determinism gate in the scoping doc).
//
// Gated by a module constant defaulting ON, mirroring the W6 recipe-fix flag
// pattern (training/r12_orchestrator.py:61). Set UMA_MCTS_KEEPALIVE=0 to fall
// back to bare fetch for an exact A/B against the pre-change transport.

import { Agent as HttpAgent, request as httpRequest } from "node:http";
import { Agent as HttpsAgent, request as httpsRequest } from "node:https";

// Throughput #4 flag. Default ON; UMA_MCTS_KEEPALIVE=0 reverts to bare fetch.
export const MCTS_KEEPALIVE_ENABLED = process.env.UMA_MCTS_KEEPALIVE !== "0";

// One shared Agent per process. keepAlive reuses sockets across the 200k+
// /predict calls instead of a TCP handshake per call. maxSockets is
// unbounded-per-host (Infinity) so concurrent in-flight requests within a
// single worker behave exactly like independent `fetch()` calls would
// (no added queuing/serialization that could perturb ordering).
const httpAgent = new HttpAgent({ keepAlive: true, maxSockets: Infinity });
const httpsAgent = new HttpsAgent({ keepAlive: true, maxSockets: Infinity });

export type PredictHttpResponse = {
  ok: boolean;
  status: number;
  text: () => Promise<string>;
  json: () => Promise<unknown>;
};

/**
 * POST a JSON body to `url` over a keep-alive connection and return a
 * fetch-Response-shaped result so call sites stay identical.
 *
 * `bodyJson` is the EXACT same string the prior code passed to
 * `fetch(..., { body: JSON.stringify(...) })`; we accept the pre-serialized
 * string so the wire bytes are guaranteed identical to the bare-fetch path.
 */
export function postJsonKeepAlive(url: string, bodyJson: string): Promise<PredictHttpResponse> {
  const parsed = new URL(url);
  const isHttps = parsed.protocol === "https:";
  const agent = isHttps ? httpsAgent : httpAgent;
  const requestFn = isHttps ? httpsRequest : httpRequest;
  const payload = Buffer.from(bodyJson, "utf8");

  return new Promise<PredictHttpResponse>((resolve, reject) => {
    const req = requestFn(
      {
        protocol: parsed.protocol,
        hostname: parsed.hostname,
        port: parsed.port,
        path: `${parsed.pathname}${parsed.search}`,
        method: "POST",
        agent,
        headers: {
          "Content-Type": "application/json",
          "Content-Length": payload.length,
        },
      },
      (res) => {
        const chunks: Buffer[] = [];
        res.on("data", (chunk: Buffer) => chunks.push(chunk));
        res.on("end", () => {
          const bodyText = Buffer.concat(chunks).toString("utf8");
          const status = res.statusCode ?? 0;
          resolve({
            ok: status >= 200 && status < 300,
            status,
            text: async () => bodyText,
            json: async () => JSON.parse(bodyText),
          });
        });
        res.on("error", reject);
      },
    );
    req.on("error", reject);
    req.end(payload);
  });
}
