import { readFile, writeFile, mkdir } from "node:fs/promises";
import { createHash } from "node:crypto";
import { request } from "node:http";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";

const root = resolve(import.meta.dirname, "..");
const snapshot = resolve(root, "build/reference-rag-pins");
const [input, report, ...options] = process.argv.slice(2);
const upstream50 = input === "--benchmark-upstream50";
const benchmark = input === "--benchmark" || upstream50;
const modelOption = options.find(x => x.startsWith("--model="))?.slice(8);
const models = ["DeepSeek-V4.1-Flash-1017", "Kimi-K3"];
const resumeDir = options.find(x => x.startsWith("--resume-dir="))?.slice(13);
const evidenceDir = options.find(x => x.startsWith("--evidence-dir="))?.slice(15);
if (evidenceDir && (!benchmark || resumeDir)) throw new Error("Evidence reuse requires a standalone benchmark.");
if (resumeDir && (!benchmark || !modelOption || !models.includes(modelOption))) throw new Error("Resume requires a benchmark and explicit model.");
const finalOutputLimits = Object.freeze({"DeepSeek-V4.1-Flash-1017":4096, "Kimi-K3":16384});
const fullEvidence = options.includes("--upstream-full-evidence");
const ruleCheckOptions = options.filter(x => x.startsWith("--rule-check="));
const extraRuleCheck = ruleCheckOptions[0]?.slice(13).trim();
if (ruleCheckOptions.length > 1 || (ruleCheckOptions.length && (!fullEvidence || !extraRuleCheck))) {
  throw new Error("A rule check requires full evidence preparation and a non-empty query.");
}
const selectedCaseIds = options.find(x => x.startsWith("--cases="))?.slice(8).split(",");
const upstreamCardExtraction = fullEvidence || options.includes("--upstream-card-extraction");
if ((fullEvidence || selectedCaseIds) && (!upstream50 || resumeDir || evidenceDir || !fullEvidence || !selectedCaseIds)) {
  throw new Error("Full evidence preparation requires a fresh upstream benchmark and explicit case IDs.");
}
const concurrency = Number(options.find(x => x.startsWith("--concurrency="))?.slice(14) ?? 2);
if (!Number.isSafeInteger(concurrency) || concurrency < 1 || concurrency > 12) throw new Error("Invalid benchmark concurrency.");
if (upstreamCardExtraction && (!upstream50 || resumeDir || evidenceDir || options.includes("--local-card-matching"))) {
  throw new Error("Upstream card extraction requires fresh upstream 50-question evidence.");
}
const localCardsOnly = (benchmark && !upstreamCardExtraction) || options.includes("--local-card-matching");
if (options.some(x => x !== "--local-card-matching" && x !== "--upstream-card-extraction" && x !== "--upstream-full-evidence" && !x.startsWith("--cases=") && !x.startsWith("--rule-check=") && !x.startsWith("--concurrency=") && !x.startsWith("--model=") && !x.startsWith("--resume-dir=") && !x.startsWith("--evidence-dir="))) throw new Error("Unknown option.");
const question = !benchmark && input?.startsWith("@") ? (await readFile(input.slice(1), "utf8")).trim() : input;
if (!question || !report) throw new Error("Usage: reference_rag.mjs QUESTION REPORT.json [--model=Kimi-K3], or --benchmark OUTPUT_DIR");
const resourceFiles = [
  "cards.json", "rulings.json", "qa-index.json", "qa-discovery-index.json",
  "evidence-index.json.gz", "ocg-rule-corpus.json", "official-responses.json",
  "rag-data-revision-manifest.json", "snapshot-meta.json",
];
const commit = "41c6b8966263796b6548cc6b368c6bc06b04dc70";
const tree = JSON.parse(await readFile(resolve(snapshot, "tree.json"), "utf8"));
if (tree.truncated) throw new Error("Incomplete upstream file inventory.");
for (const file of resourceFiles) {
  const name = "data/" + file;
  const entry = tree.tree.find(row => row.path === name);
  const bytes = await readFile(resolve(snapshot, "verified-data", file));
  const actual = createHash("sha1").update("blob " + bytes.length + "\0").update(bytes).digest("hex");
  if (!entry || entry.sha !== actual || entry.size !== bytes.length) {
    throw new Error("Missing or mismatched upstream resource: " + name);
  }
}
// Verify the pinned JavaScript import set before loading server credentials.
for (const name of (await readFile(resolve(snapshot, "code-files.txt"), "utf8")).trim().split(/\r?\n/u)) {
  const entry = tree.tree.find(row => row.path === name);
  const bytes = await readFile(resolve(snapshot, name));
  const actual = createHash("sha1").update("blob " + bytes.length + "\0").update(bytes).digest("hex");
  if (!entry || entry.sha !== actual) throw new Error("Upstream code mismatch: " + name);
}
const corpusPointer = resolve(root, "build/reference-rag-corpora/current.json");
const readJson = async path => JSON.parse(await readFile(path, "utf8"));
const priorDir = resumeDir || evidenceDir;
let selectedCorpus;
try {
  selectedCorpus = priorDir ? (await readJson(resolve(priorDir, "plan.json"))).corpus
    : process.env.UPSTREAM_OCG_PINNED_CORPUS === "1" ? undefined : await readJson(corpusPointer);
} catch (error) {
  if (error.code !== "ENOENT" || priorDir) throw error;
}
const corpus = selectedCorpus || { commit, dataDir: "build/reference-rag-pins/verified-data" };
if (!/^[a-f0-9]{40}$/u.test(corpus.commit)
    || !["build/reference-rag-pins/verified-data", "build/reference-rag-corpora/" + corpus.commit].includes(corpus.dataDir)) {
  throw new Error("Unexpected upstream corpus directory");
}
const dataDir = resolve(root, corpus.dataDir);
const manifest = await readJson(resolve(dataDir, "rag-data-revision-manifest.json"));
if (corpus.revision && corpus.revision !== manifest.revision) throw new Error("Upstream corpus revision mismatch");
corpus.revision = manifest.revision;
const meta = await readJson(resolve(dataDir, "snapshot-meta.json"));
process.loadEnvFile(resolve(root, ".env"));
const gateway = String(process.env.DEEPSEEK_BASE_URL || "").replace(/\/+$/u, "");
const key = process.env.DEEPSEEK_API_KEY;
const model = modelOption || process.env.DEEPSEEK_MODEL;
if (gateway !== "http://117.135.59.13:4000/v1" || !models.includes(model) || !key) {
  throw new Error("Configured project gateway, model or credential is unavailable.");
}
const timeoutMs = 1000 * Number(process.env.DEEPSEEK_TIMEOUT_SECONDS || 600);
if (!Number.isFinite(timeoutMs) || timeoutMs < 1000) throw new Error("Invalid model timeout.");
const calls = [];
const started = Date.now();
const makeInvoker = (model, calls, expectedPromptSha) => async ({ prompt, maxTokens, task, signal }) => {
  const promptSha256 = createHash("sha256").update(prompt).digest("hex");
  if (expectedPromptSha && promptSha256 !== expectedPromptSha) throw new Error("Frozen prompt mismatch.");
  // K3 can spend the response budget on reasoning before writing any answer.
  const outputLimit = !task && model === "Kimi-K3" ? finalOutputLimits[model]
    : Math.min(maxTokens || 4096, 4096);
  const call = {maxOutputTokens:outputLimit, stage: task || "final_ruling", status:"running", promptChars:prompt.length, promptSha256, startedAt:new Date().toISOString()};
  calls.push(call);
  console.error(call.stage + " started");
  try {
    const timeout = AbortSignal.timeout(timeoutMs);
    const body = JSON.stringify({model, messages:[{role:"user",content:prompt}], temperature:0,
      max_tokens:outputLimit, thinking:{type:"disabled"}});
    const payload = await new Promise((accept, reject) => {
      const req = request(gateway + "/chat/completions", {
        method:"POST", signal:signal ? AbortSignal.any([signal, timeout]) : timeout,
        headers:{"content-type":"application/json", "content-length":Buffer.byteLength(body),
          authorization:"Bearer " + key},
      }, (response) => {
        if (response.statusCode < 200 || response.statusCode >= 300) {
          response.resume(); reject(new Error("Gateway HTTP " + response.statusCode)); return;
        }
        response.setEncoding("utf8");
        let received = "";
        response.on("data", chunk => { received += chunk; });
        response.on("error", reject);
        response.on("end", () => {
          try { accept(JSON.parse(received)); } catch (error) { reject(error); }
        });
      });
      req.on("error", reject);
      req.end(body);
    });
    const choice = payload.choices?.[0];
    call.responseModel = payload.model;
    call.finishReason = choice?.finish_reason;
    call.usage = payload.usage;
    if (!choice?.message?.content?.trim() || choice.finish_reason === "length") {
      throw new Error("Empty or truncated response.");
    }
    if (payload.model !== model) throw new Error("Unexpected response model.");
    call.status = "completed";
    if (task === "rule_query_extraction" && extraRuleCheck) {
      const content = choice.message.content.trim().replace(/^```(?:json)?\s*|\s*```$/giu, "");
      const plan = JSON.parse(content);
      const field = ["ruleQueries", "queries", "ruleSearchQueries", "keywords"].find(name => Array.isArray(plan[name]));
      if (!field) throw new Error("Rule query model returned no query list.");
      plan[field].unshift({
        subclaim: extraRuleCheck,
        checkpoint: "activation_snapshot",
        officialQuestion: extraRuleCheck,
        scenarioQuestion: extraRuleCheck,
        reason: "Explicit independent rule check supplied with the benchmark.",
        confidence: "high",
      });
      return plan;
    }
    return choice.message.content;
  } catch (error) {
    call.status = "failed";
    call.errorType = error.name;
    call.errorCode = error.code;
    call.httpStatus = /^Gateway HTTP (\d+)$/.exec(error.message)?.[1];
    throw error;
  }
};
const makeEnv = model => ({RAG_MODEL_PROVIDER:"deepseek",RAG_CARD_MODEL_PROVIDER:"deepseek",
  RAG_RULE_QUERY_EXTRACTOR_ENABLED:fullEvidence ? "true" : "false", DEEPSEEK_BASE_URL:gateway,DEEPSEEK_MODEL:model,
  DEEPSEEK_CARD_MODEL:model,DEEPSEEK_RULE_MODEL:model,DEEPSEEK_API_KEY:key,
  ...(localCardsOnly ? {RAG_CARD_EXTRACTOR_ENABLED:"false"} : {}),
  RAG_LIVE_OFFICIAL_QA:fullEvidence ? "true" : "false",RAG_EVIDENCE_APPLICABILITY_ENABLED:"false",
  // Activate the pinned local pipeline's Relay planner. The injected invoker
  // handles transport, so this sentinel endpoint and value are never sent.
  ...(fullEvidence ? {RAG_RULE_MODEL_PROVIDER:"relay", RELAY_BASE_URL:"https://relay.invalid/v1",
    RELAY_API_KEY:"injected-model-invoker"} : {}),
});
const fetchPublicQa = async (url, options) => {
  const target = new URL(url);
  if (target.origin !== "https://db.ygoresources.com" || target.username || target.password) {
    throw new Error("Unexpected evidence source.");
  }
  return fetch(target, {...options, redirect:"error"});
};
const noOtherNetwork = async () => { throw new Error("External runtime provider disabled in this local route."); };
const {answerRagRulingQuestion, finalizePreparedRagRulingQuestion} = await import(
  pathToFileURL(resolve(snapshot,"backend/ragRulingPipeline.mjs")));
const save = (path, data) => writeFile(path, JSON.stringify(data,null,2).replaceAll(key,"[REDACTED]")+"\n",
  {encoding:"utf8",flag:"wx"});

if (benchmark) {
  const bankPath = upstream50 ? resolve(snapshot,"ocg-50-20260924.json")
    : resolve(root,"benchmarks/ocgrule_acceptance_v2/questions.json");
  const bytes = await readFile(bankPath);
  const bank = JSON.parse(bytes.toString("utf8"));
  if (upstream50) {
    const entry = tree.tree.find(row=>row.path==="scripts/bench/sets/ocg-50-20260924.json");
    const blob = createHash("sha1").update("blob "+bytes.length+"\0").update(bytes).digest("hex");
    if (!entry || entry.sha!==blob) throw new Error("Upstream 50-question snapshot mismatch.");
  }
  const bankCases = upstream50 ? bank.items.map(({qid,question})=>({case_id:qid,query:question})) : bank.cases;
  if (bankCases.length !== (upstream50 ? 50 : 24) || new Set(bankCases.map(c=>c.case_id)).size!==bankCases.length
      || bankCases.some(c=>!/^[a-zA-Z0-9-]+$/u.test(c.case_id) || !c.query?.trim())) {
    throw new Error("Invalid frozen panel.");
  }
  const cases = selectedCaseIds ? bankCases.filter(c => selectedCaseIds.includes(c.case_id)) : bankCases;
  if (selectedCaseIds && (cases.length !== selectedCaseIds.length || !cases.length)) throw new Error("Invalid case IDs.");
  const selectedModels = modelOption ? [model] : models;
  const readOptional = async path => {
    try { return await readFile(path); }
    catch (error) { if (error.code === "ENOENT") return null; throw error; }
  };
  const benchmarkSha256 = createHash("sha256").update(bytes).digest("hex");
  if (resumeDir) {
    if (resolve(resumeDir) === resolve(report)) throw new Error("Resume requires a new output directory.");
    const prior = JSON.parse(await readFile(resolve(resumeDir,"plan.json"),"utf8"));
    if (prior.benchmarkSha256 !== benchmarkSha256 || prior.commit !== commit
        || prior.sourceRevision !== meta.sourceRevision || prior.thinking !== "disabled"
        || prior.temperature !== 0 || prior.timeoutMs !== timeoutMs
        || !prior.models.includes(model)
        || (prior.finalOutputLimits?.[model] ?? prior.maxTokens) !== finalOutputLimits[model]) {
      throw new Error("Resume configuration differs from original run.");
    }
  }
  if (evidenceDir) {
    const prior = JSON.parse(await readFile(resolve(evidenceDir,"plan.json"),"utf8"));
    if (prior.benchmarkSha256 !== benchmarkSha256 || prior.commit !== commit
        || prior.sourceRevision !== meta.sourceRevision
        || JSON.stringify(prior.cases) !== JSON.stringify(cases.map(({case_id,query})=>({case_id,query})))) {
      throw new Error("Evidence belongs to a different frozen run.");
    }
    const terminal = JSON.parse(await readFile(resolve(evidenceDir,"terminal.json"),"utf8"));
    if (terminal.finished !== cases.length) throw new Error("Evidence run is incomplete.");
  }
  await mkdir(report);
  await save(resolve(report,"plan.json"), {
    benchmarkId:upstream50 ? bank.setId : bank.catalogue_version,
    upstreamGoldExcludedFromModelInput:true, benchmarkSha256, commit, corpus, sourceRevision:meta.sourceRevision,
    models:selectedModels, resumeDir, evidenceDir, timeoutMs, thinking:"disabled", temperature:0, finalOutputLimits,
    concurrency, extraRuleCheck, preparationOnly:fullEvidence, liveOfficialQa:fullEvidence, ruleQueryTransport:fullEvidence ? selectedModels[0] : null, cardExtraction:upstreamCardExtraction ? "upstream_model_and_candidate_confirmation" : "upstream_local_matching",
    cardExtractionModel:upstreamCardExtraction ? selectedModels[0] : null,
    protocol:"Prepare the upstream continuation once per question and reuse it across models. No scoring rubric is provided to model calls.",
    cases:cases.map(({case_id,query})=>({case_id,query})),
  });
  let finished = 0, reused = 0, unavailable = 0;
  // Keep preparation serial to avoid parallel cold-start index builds. Both
  // contenders can generate concurrently on each prepared pair.
  const active = new Set();
  const prepareCase = async c => {
    const caseId = c.case_id;
    const prepName = caseId+"-prepare.json";
    const previous = evidenceDir ? await readFile(resolve(evidenceDir,prepName))
      : resumeDir && await readOptional(resolve(resumeDir,prepName));
    const preparationCalls = [];
    let prepared;
    try {
      prepared = previous ? {status:"evidence_prepared",continuation:JSON.parse(previous)}
        : await answerRagRulingQuestion({question:c.query, dataDir, env:makeEnv(selectedModels[0]),
          cardModelInvoker:upstreamCardExtraction ? makeInvoker(selectedModels[0], preparationCalls) : undefined,
          ruleModelInvoker:fullEvidence ? makeInvoker(selectedModels[0], preparationCalls) : undefined,
          prepareForContinuation:true, fetchImpl:fullEvidence ? fetchPublicQa : noOtherNetwork, thinkingMode:"disabled"});
    } catch (error) {
      if (!upstreamCardExtraction) throw error;
      prepared = {status:"preparation_failed", failure:error.name + ": " + error.message};
    }
    if (upstreamCardExtraction) {
      await save(resolve(report,caseId+"-card-extraction.json"),{preparationCalls,status:prepared.status});
    }
    if (prepared.status !== "evidence_prepared") {
      await save(resolve(report,caseId+"-prepare.json"),{status:prepared.status, answerLevel:prepared.answerLevel,
        shortAnswer:prepared.shortAnswer, missingInfo:prepared.missingInfo, failure:prepared.failure});
      console.log(caseId+" no common final prompt: "+(prepared.status || prepared.answerLevel));
      unavailable += selectedModels.length;
      return null;
    }
    const continuation = prepared.continuation;
    if (!continuation.finalPromptSha256 || createHash("sha256").update(continuation.promptBundle.prompt).digest("hex") !== continuation.finalPromptSha256) {
      throw new Error("Invalid prepared prompt: "+caseId);
    }
    await save(resolve(report,prepName),continuation);
    return continuation;
  };
  const answerCase = (caseId, continuation) => Promise.all(selectedModels.map(async selected => {
      const outputName = caseId+"-"+selected+".json";
      const old = resumeDir && await readOptional(resolve(resumeDir,outputName));
      if (old) {
        const prior = JSON.parse(old);
        if (prior.caseId !== caseId || prior.model !== selected || prior.promptSha256 !== continuation.finalPromptSha256) {
          throw new Error("Old output does not match frozen question/model/prompt: "+caseId);
        }
        await writeFile(resolve(report,outputName),old,{flag:"wx"});
        reused++; finished++;
        console.log(caseId+" "+selected+" retained");
        return;
      }
      const calls = [];
      let result, failure;
      try {
        result = await finalizePreparedRagRulingQuestion({continuation, env:makeEnv(selected),
          modelInvoker:makeInvoker(selected,calls,continuation.finalPromptSha256),
          thinkingMode:"disabled", fetchImpl:noOtherNetwork});
        if (!calls.some(c => c.status === "completed")) failure = "No final model output";
      } catch(error) { failure = error.name+": "+error.message; }
      await save(resolve(report,outputName),{caseId,model:selected,
        promptSha256:continuation.finalPromptSha256,calls,result,failure});
      console.log(caseId+" "+selected+" "+(failure ? "failed" : "completed"));
      finished++;
  }));
  if (upstreamCardExtraction) {
    let next = 0;
    await Promise.all(Array.from({length:concurrency}, async () => {
      while (next < cases.length) {
        const item = cases[next++];
        const continuation = await prepareCase(item);
        if (continuation && fullEvidence) { finished++; console.log(item.case_id+" prepared"); }
        if (continuation && !fullEvidence) await answerCase(item.case_id, continuation);
      }
    }));
  } else {
    for (const item of cases) {
      const continuation = await prepareCase(item);
      if (!continuation) continue;
      const pair = answerCase(item.case_id, continuation);
      active.add(pair);
      pair.then(()=>active.delete(pair),()=>{});
      if (active.size >= concurrency) await Promise.race(active);
    }
    await Promise.all(active);
  }
  await save(resolve(report,"terminal.json"),{finished,reused,fresh:finished-reused,unavailable});
  process.exit(0);
}

const env = makeEnv(model);
const invoke = makeInvoker(model,calls);
let result, failure;
try {
  result = await answerRagRulingQuestion({question, dataDir:resolve(snapshot,"verified-data"), env,
    cardModelInvoker:localCardsOnly ? undefined : invoke, modelInvoker:invoke,
    fetchImpl:noOtherNetwork, thinkingMode:"disabled"});
  if (!calls.some(c => c.stage === "final_ruling" && c.status === "completed")) {
    failure = "No final model response; inspect upstream status.";
  }
} catch (error) {
  failure = error.name + ": " + error.message;
}
const record = JSON.stringify({commit, sourceRevision:meta.sourceRevision, snapshotAt:meta.generatedAt,
  route:"upstream local-evidence pipeline (deterministic search hints)", model, timeoutMs, elapsedMs:Date.now()-started,
  question,localCardsOnly,finalOutputLimit:finalOutputLimits[model],calls,result,failure},null,2).replaceAll(key,"[REDACTED]");
await writeFile(report, record+"\n", {encoding:"utf8",flag:"wx"});
console.log(JSON.stringify({report,calls:calls.map(c=>[c.stage,c.status]),resultKeys:Object.keys(result || {}),failure}));
if (failure) process.exitCode=1;
