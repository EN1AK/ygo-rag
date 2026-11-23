import { createHash } from "node:crypto";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { createInterface } from "node:readline";
import { Readable } from "node:stream";
import { retrieveRagEvidence } from "../build/reference-rag-pins/backend/ragEvidenceRetriever.mjs";
import { buildRagRulingPromptBundle } from "../build/reference-rag-pins/backend/ragRulingPrompt.mjs";
import { finalizePreparedRagRulingQuestion } from "../build/reference-rag-pins/backend/ragRulingPipeline.mjs";

const [source, destination, ...caseIds] = process.argv.slice(2);
if (!source || !destination || !caseIds.length || caseIds.some(id => !/^Q[1-9][0-9]*$/u.test(id))
    || new Set(caseIds).size !== caseIds.length) {
  throw new Error("Usage: reference_rag_loop.mjs PREPARED_DIR NEW_OUTPUT_DIR QID [QID ...]");
}

const root = resolve(import.meta.dirname, "..");
const sha = value => createHash("sha256").update(value).digest("hex");
const load = async path => JSON.parse(await readFile(path, "utf8"));
const sourcePlan = await load(resolve(source, "plan.json"));
if (sourcePlan.cardExtraction !== "upstream_model_and_candidate_confirmation"
    || sourcePlan.commit !== "41c6b8966263796b6548cc6b368c6bc06b04dc70") {
  throw new Error("Expected the full-evidence run from the pinned upstream revision");
}
const defaultDataDir = "build/reference-rag-pins/verified-data";
const corpus = sourcePlan.corpus || { commit: sourcePlan.commit, dataDir: defaultDataDir };
if (!/^[a-f0-9]{40}$/u.test(corpus.commit)
    || ![defaultDataDir, "build/reference-rag-corpora/" + corpus.commit].includes(corpus.dataDir)) {
  throw new Error("Unexpected frozen upstream corpus directory");
}
const dataDir = resolve(root, corpus.dataDir);
const { revision } = await load(resolve(dataDir, "rag-data-revision-manifest.json"));
if (corpus.revision && revision !== corpus.revision) throw new Error("Frozen upstream corpus revision mismatch");
const frozen = await Promise.all(caseIds.map(async id => {
  const continuation = await load(resolve(source, id + "-prepare.json"));
  if (continuation.mode !== "rag_baseline" || continuation.schemaVersion !== 1
      || (sourcePlan.corpus && continuation.dataRevision !== revision)
      || sha(continuation.promptBundle.prompt) !== continuation.finalPromptSha256
      || !sourcePlan.cases.some(item => item.case_id === id && item.query === continuation.promptBundle.promptPayload.userQuery)) {
    throw new Error(id + " frozen upstream continuation mismatch");
  }
  return [id, continuation];
}));

process.loadEnvFile(resolve(root, ".env"));
const base = process.env.DEEPSEEK_BASE_URL?.replace(/\/+$/u, "");
const model = process.env.DEEPSEEK_MODEL;
const key = process.env.DEEPSEEK_API_KEY;
if (base !== "http://117.135.59.13:4000/v1" || model !== "DeepSeek-V4.1-Flash-1017" || !key) {
  throw new Error("Project DeepSeek configuration unavailable");
}
const timeoutMs = Math.max(1800, Number(process.env.DEEPSEEK_TIMEOUT_SECONDS || 600)) * 1000;
// Keep cloud providers, live Q&A, and the hosted-only supplement out of this local upstream run.
const upstreamEnv = { RAG_MODEL_PROVIDER: "deepseek", DEEPSEEK_MODEL: model, DEEPSEEK_BASE_URL: base,
  DEEPSEEK_API_KEY: key, RAG_LIVE_OFFICIAL_QA: "false", RAG_EVIDENCE_APPLICABILITY_ENABLED: "false" };
const noOtherNetwork = async () => { throw new Error("Only the configured DS transport can use the network"); };

await mkdir(destination);
const save = (name, value) => writeFile(resolve(destination, name), JSON.stringify(value, null, 2) + "\n", "utf8");
await save("plan.json", { model, timeoutMs, finalReasoningEffort: "max", auditThinking: "disabled",
  finalOutputTokens: 65536, auditOutputTokens: 2048, supplementLimit: 1, upstreamCommit: sourcePlan.commit, sourceRevision: sourcePlan.sourceRevision, corpus,
  dataRevisions: [...new Set(frozen.map(([, item]) => item.dataRevision))], source, caseIds });

async function invoke({ id, stage, prompt, calls, signal, audit = false }) {
  const call = { stage, promptSha256: sha(prompt), promptChars: prompt.length, status: "running",
    thinking: audit ? "disabled" : "max" };
  calls.push(call);
  const start = Date.now();
  console.error(id + " " + stage + " started");
  try {
    const timeout = AbortSignal.timeout(timeoutMs);
    const response = await fetch(base + "/chat/completions", {
      method: "POST",
      headers: { authorization: "Bearer " + key, "content-type": "application/json" },
      signal: signal ? AbortSignal.any([signal, timeout]) : timeout,
      body: JSON.stringify({ model, messages: [{ role: "user", content: prompt }], temperature: 0,
        max_tokens: audit ? 2048 : 65536, reasoning_effort: audit ? undefined : "max",
        thinking: { type: audit ? "disabled" : "enabled" }, stream: true,
        stream_options: { include_usage: true } }),
    });
    if (!response.ok) {
      await response.body?.cancel();
      throw new Error("Gateway HTTP " + response.status);
    }
    let content = "";
    const lines = createInterface({ input: Readable.fromWeb(response.body), crlfDelay: Infinity });
    for await (const line of lines) {
      if (!line.startsWith("data:")) continue;
      const data = line.slice(5).trim();
      if (data === "[DONE]") break;
      const packet = JSON.parse(data);
      if (packet.error) throw new Error("Gateway reported a streaming error");
      if (packet.model) call.responseModel = packet.model;
      if (packet.usage) call.usage = packet.usage;
      const choice = packet.choices?.[0];
      if (choice?.finish_reason) call.finishReason = choice.finish_reason;
      content += choice?.delta?.content || "";
    }
    if (call.responseModel !== model || call.finishReason !== "stop" || !content.trim()) {
      throw new Error("Gateway returned an incomplete answer or unexpected model");
    }
    call.status = "completed";
    return content;
  } catch (error) {
    call.status = "failed";
    throw error;
  } finally {
    call.elapsedMs = Date.now() - start;
    console.error(id + " " + stage + " " + call.status + " in " + call.elapsedMs + " ms");
  }
}

async function finalize(id, stage, continuation, calls) {
  const result = await finalizePreparedRagRulingQuestion({
    continuation, env: upstreamEnv, fetchImpl: noOtherNetwork,
    modelInvoker: ({ prompt, signal }) => {
      if (sha(prompt) !== continuation.finalPromptSha256) throw new Error("Final prompt mismatch");
      return invoke({ id, stage, prompt, signal, calls });
    },
  });
  if (!result.shortAnswer?.trim() || calls.at(-1)?.status !== "completed") {
    throw new Error("Upstream finalization did not yield an answer");
  }
  return result;
}

async function audit(id, continuation, answer, calls) {
  const prompt = [
    "检查已提交的游戏王 OCG 裁定答复是否完整、有直接适用的证据。此步骤只决定是否再检索，不作答。",
    "逐一核对用户要求的结论和独立时点。草稿语气确定也不代表有证据。区分「用户缺少局面信息」和「证据缺少可检索的规则」，前者不触发补查。",
    "仅输出 JSON 对象 {gap,queries}。证据足够或缺少局面信息时 gap 为空字符串、queries 为空数组。",
    "发现实质性规则缺口时用中文概述 gap，给 1 至 3 条查询，每条包含 subclaim、checkpoint、officialQuestion（日语官方问法）、scenarioQuestion（完整问题，可用中文）。",
    "查询应单独核实决定性规则、适用例外或相反情形，不猜答案，不把搜索词当成新局面事实。以下为待检查数据。",
    JSON.stringify({ draft: answer }),
    continuation.promptBundle.prompt,
  ].join("\n");
  const raw = await invoke({ id, stage: "coverage_audit", prompt, calls, audit: true });
  const plan = JSON.parse(raw.trim().replace(/^```(?:json)?\s*|\s*```$/giu, ""));
  if (typeof plan?.gap !== "string" || !Array.isArray(plan.queries) || plan.queries.length > 3) {
    throw new Error("Invalid coverage audit result");
  }
  const fields = ["subclaim", "checkpoint", "officialQuestion", "scenarioQuestion"];
  const queries = plan.queries.map(item => Object.fromEntries(fields.map(field => {
    if (typeof item?.[field] !== "string" || !item[field].trim()) throw new Error("Invalid audit query");
    return [field, item[field].trim().slice(0, 220)];
  })));
  const gap = plan.gap.trim().slice(0, 600);
  if (Boolean(gap) !== Boolean(queries.length)) throw new Error("Inconsistent coverage audit result");
  return { gap, queries };
}

function mergeUpstreamEvidence(original, retrieved) {
  const evidence = { ...original, ruleSearchQueries: retrieved.ruleSearchQueries };
  for (const bucket of ["officialQaRelated", "provisionalOfficialResponses", "faqRelated",
    "rawRelatedEvidence", "rulebookCandidates"]) {
    const candidates = [...(retrieved[bucket] || []), ...(original[bucket] || [])];
    evidence[bucket] = [...new Map(candidates.reverse().map(item => [item.id, item])).values()].reverse();
  }
  return evidence;
}

async function runCase([id, original]) {
  const report = { id, question: original.promptBundle.promptPayload.userQuery, firstPromptSha256: original.finalPromptSha256,
    originalEvidenceIds: original.promptBundle.allowedEvidenceIds, calls: [] };
  try {
    report.baseline = await finalize(id, "baseline", original, report.calls);
    await save(id + "-result.json", report);
    report.audit = await audit(id, original, report.baseline.shortAnswer, report.calls);
    if (!report.audit.queries.length) {
      report.status = "no_gap";
      return report;
    }
    console.error(id + " upstream retrieval started");
    const retrievalStart = Date.now();
    const retrieved = await retrieveRagEvidence({
      userQuery: report.question, cardResolution: original.cardResolution,
      ruleSearchQueries: [...report.audit.queries, ...(original.evidence.ruleSearchQueries || [])], dataDir,
      env: upstreamEnv, fetchImpl: noOtherNetwork,
    });
    report.retrievalMs = Date.now() - retrievalStart;
    const evidence = mergeUpstreamEvidence(original.evidence, retrieved);
    const promptBundle = buildRagRulingPromptBundle({
      userQuery: report.question, answerLocale: original.answerLocale,
      cardResolution: original.cardResolution, evidence, env: upstreamEnv,
    });
    const originalIds = new Set(report.originalEvidenceIds);
    const nextIds = new Set(promptBundle.allowedEvidenceIds);
    report.addedEvidenceIds = [...nextIds].filter(id => !originalIds.has(id));
    report.removedEvidenceIds = [...originalIds].filter(id => !nextIds.has(id));
    report.effectiveQueries = retrieved.ruleSearchQueries;
    report.rulebookCandidates = retrieved.rulebookCandidates.map(item => ({ id: item.id, title: item.title }));
    if (!report.addedEvidenceIds.length) {
      report.status = "no_new_admitted_evidence";
      return report;
    }
    const { evidenceFingerprint: frozenEvidenceFingerprint, ...metadata } = original;
    report.originalEvidenceFingerprint = frozenEvidenceFingerprint;
    const next = { ...metadata, evidence, promptBundle,
      finalPromptSha256: sha(promptBundle.prompt) };
    await save(id + "-supplement-prompt.json", promptBundle);
    report.supplement = await finalize(id, "supplement", next, report.calls);
    report.status = "applied";
  } catch (error) {
    report.status = report.baseline ? "kept_baseline_after_failure" : "baseline_failed";
    report.error = error.message;
  } finally {
    await save(id + "-result.json", report);
    console.error(id + " " + report.status);
  }
  return report;
}

const results = await Promise.all(frozen.map(runCase));
await save("terminal.json", results.map(({ id, status }) => ({ id, status })));
if (results.some(({ status }) => status.endsWith("failure") || status.endsWith("failed"))) process.exitCode = 1;
