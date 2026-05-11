import { AsyncLocalStorage } from "node:async_hooks";
import { installRngStorageProvider, type Rng } from "../../../frontend/src/game/engine/core/random";

const storage = new AsyncLocalStorage<Rng>();

installRngStorageProvider({
  get: () => storage.getStore() ?? null,
  run: <T>(rng: Rng, fn: () => T): T => storage.run(rng, fn),
});
