import assert from "node:assert/strict";
import { runHeadlessBatch } from "../sim/headlessAiVsAi";
import type { TrainingExample } from "../../../frontend/src/game/engine/ai-policy/types";

const FEATURE_COUNT = 18;
const EPOCHS = 35;
const LEARNING_RATE = 0.18;

const runs = runHeadlessBatch(["101", "202", "303", "404", "505", "606", "707", "808"], 360);
const examples = runs.flatMap((run) => run.examples).filter((example) => example.legalActions.length > 1);

assert.ok(runs.length === 8, "expected eight deterministic headless runs");
assert.ok(runs.every((run) => run.terminalReason === "gameOver"), `all runs should finish, got ${runs.map((run) => run.terminalReason).join(", ")}`);
assert.ok(examples.length >= 40, `expected enough training rows, got ${examples.length}`);
assertNoHiddenOpponentHand(examples);

const model = createLinearModel(FEATURE_COUNT);
const initialLoss = averageLoss(model, examples);
for (let epoch = 0; epoch < EPOCHS; epoch += 1) {
  for (const example of examples) {
    trainOne(model, example, LEARNING_RATE);
  }
}
const finalLoss = averageLoss(model, examples);
const accuracy = topOneAccuracy(model, examples);

assert.ok(Number.isFinite(initialLoss), "initial loss should be finite");
assert.ok(Number.isFinite(finalLoss), "final loss should be finite");
assert.ok(finalLoss < initialLoss * 0.82, `loss should drop enough: initial=${initialLoss}, final=${finalLoss}`);
assert.ok(accuracy >= 0.72, `training top-1 accuracy should be useful, got ${accuracy}`);

const repeat = runHeadlessBatch(["101"], 360)[0];
assert.deepEqual(
  compactRun(runs[0]!),
  compactRun(repeat!),
  "same seed should reproduce the same compact simulator result",
);

console.log(JSON.stringify({
  status: "PASS",
  runs: runs.length,
  examples: examples.length,
  initialLoss: Number(initialLoss.toFixed(4)),
  finalLoss: Number(finalLoss.toFixed(4)),
  topOneAccuracy: Number(accuracy.toFixed(4)),
  winners: runs.map((run) => run.winner),
}, null, 2));

type LinearModel = {
  weights: number[];
};

function createLinearModel(size: number): LinearModel {
  return { weights: Array.from({ length: size }, () => 0) };
}

function trainOne(model: LinearModel, example: TrainingExample, learningRate: number): void {
  const probabilities = softmax(example.legalActions.map((action) => dot(model.weights, action.features)));
  for (let actionIndex = 0; actionIndex < example.legalActions.length; actionIndex += 1) {
    const action = example.legalActions[actionIndex]!;
    const target = actionIndex === example.selectedActionIndex ? 1 : 0;
    const error = target - (probabilities[actionIndex] ?? 0);
    for (let featureIndex = 0; featureIndex < model.weights.length; featureIndex += 1) {
      model.weights[featureIndex] = (model.weights[featureIndex] ?? 0) + learningRate * error * (action.features[featureIndex] ?? 0);
    }
  }
}

function averageLoss(model: LinearModel, examples: TrainingExample[]): number {
  const total = examples.reduce((sum, example) => {
    const probabilities = softmax(example.legalActions.map((action) => dot(model.weights, action.features)));
    return sum - Math.log(Math.max(1e-8, probabilities[example.selectedActionIndex] ?? 1e-8));
  }, 0);
  return total / examples.length;
}

function topOneAccuracy(model: LinearModel, examples: TrainingExample[]): number {
  const correct = examples.reduce((sum, example) => {
    const scores = example.legalActions.map((action) => dot(model.weights, action.features));
    const bestIndex = scores.reduce((best, score, index) => score > (scores[best] ?? Number.NEGATIVE_INFINITY) ? index : best, 0);
    return sum + (bestIndex === example.selectedActionIndex ? 1 : 0);
  }, 0);
  return correct / examples.length;
}

function softmax(scores: number[]): number[] {
  const max = Math.max(...scores);
  const exps = scores.map((score) => Math.exp(score - max));
  const total = exps.reduce((sum, value) => sum + value, 0);
  return exps.map((value) => value / total);
}

function dot(left: number[], right: number[]): number {
  return left.reduce((sum, value, index) => sum + value * (right[index] ?? 0), 0);
}

function assertNoHiddenOpponentHand(examplesToCheck: TrainingExample[]): void {
  for (const example of examplesToCheck) {
    assert.equal(example.observation.opponent.handCardIds, undefined, "opponent hand IDs must not be exposed");
    assert.equal(example.observation.opponent.handCount >= 0, true, "opponent hand count should be exposed");
  }
}

function compactRun(run: NonNullable<ReturnType<typeof runHeadlessBatch>[number]>) {
  return {
    seed: run.seed,
    terminalReason: run.terminalReason,
    winner: run.winner,
    points: run.points,
    steps: run.steps,
    turnNumber: run.turnNumber,
    finalLog: run.finalLog.slice(0, 4),
  };
}
