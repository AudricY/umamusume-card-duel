import type { EnergyType } from "../../../../../shared/src/types";

export type Rng = {
  next: () => number;
  fork: (label: string) => Rng;
  label?: string;
};

let activeRng: Rng | null = null;

export function createSeededRng(seed: string | number, label = "root"): Rng {
  const initial = normalizeSeed(seed);
  let state = initial || 0x6d2b79f5;
  return {
    label,
    next: () => {
      state += 0x6d2b79f5;
      let value = state;
      value = Math.imul(value ^ (value >>> 15), value | 1);
      value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
      return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
    },
    fork: (forkLabel: string) => createSeededRng(`${initial}:${label}:${forkLabel}`, `${label}/${forkLabel}`),
  };
}

export function withRng<T>(rng: Rng, run: () => T): T {
  const previous = activeRng;
  activeRng = rng;
  try {
    return run();
  } finally {
    activeRng = previous;
  }
}

export function randomFloat(): number {
  return activeRng?.next() ?? Math.random();
}

export function randomInt(maxExclusive: number): number {
  if (maxExclusive <= 0) return 0;
  return Math.floor(randomFloat() * maxExclusive);
}

export function shuffle<T>(items: T[]): T[] {
  const copy = [...items];
  for (let index = copy.length - 1; index > 0; index -= 1) {
    const swapIndex = randomInt(index + 1);
    const current = copy[index];
    const swap = copy[swapIndex];
    if (current === undefined || swap === undefined) continue;
    copy[index] = swap;
    copy[swapIndex] = current;
  }
  return copy;
}

export function rollEnergyFromPool(pool: EnergyType[]): EnergyType {
  const index = randomInt(pool.length);
  return pool[index] ?? "psychic";
}

function normalizeSeed(seed: string | number): number {
  if (typeof seed === "number" && Number.isFinite(seed)) return seed >>> 0;
  const text = String(seed);
  let hash = 2166136261;
  for (let index = 0; index < text.length; index += 1) {
    hash ^= text.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}
