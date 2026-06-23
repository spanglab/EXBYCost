# EXBYCost — Development Plan: Migrating to a Deepnote App with Pre-batched, Cache-backed TEA

> **Status:** Proposed (not yet implemented).
> **Scope:** Migrate the existing Streamlit TEA tool to a hosted Deepnote app, and
> re-architect the parameter sweep around a pre-batched physical-simulation cache
> with a live (instant) economic layer.

---

## 0. Goals & constraints

These shape every decision below:

- **Free to host.** No paid Streamlit/server tier. Target = Deepnote on the **GitHub Student
  Developer Pack** (free hardware for ~2 years), which has native Streamlit support.
- **Easy for non-expert users.** The audience is lab researchers comfortable with light Python,
  not software engineers. The hosted tool must be usable in a browser with zero setup — no
  GitHub/clone/CLI steps for end users.
- **Multi-user friendly.** Several people should be able to use it at once without each one
  triggering a slow recompute. This is why results are **pre-computed once and read from a
  cache** rather than simulated per request.
- **Fast & correctable.** Today the sweep runs **serially** and is the core pain point: ~200+
  data points took *days*, and a single mistake forced a **full redo**. The new design must be
  (a) bounded — run the heavy grid **once**; (b) **parallel** — use all available CPU cores so
  the one-time batch is hours, not days (see §3.1); (c) **resumable** — a crash or disconnect
  never loses completed work; and (d) **correctable** — fixing one input recomputes only what
  changed, never the whole batch.
- **Cover all products.** The app currently ships only **4 feedstocks**, but more product
  datasets exist (data lives in per-product Google Sheets). Scope includes folding in the full
  product list, and the pre-batch grid must cover **every** product, not just the original four.

### Prerequisites (do these before any code work)

1. **Get Deepnote access (free).** Sign up for the **GitHub Student Developer Pack** (verify with
   a transcript / student ID / acceptance or financial-aid letter; approval ~30 min, then up to
   ~24 h). Then activate the bundle and log in to Deepnote via GitHub → free hardware for ~2 yr.
   *Also check first whether the institution already has a Deepnote account/relationship* — that
   may be simpler than an individual signup.
2. **Provide the external API keys** used by `pricing.py` — `FRED_API_KEY`, `EIA_API_KEY`,
   `BLS_API_KEY` — to be set as **Deepnote environment variables**. 🚫 **Never commit these
   keys to the repo.**
3. **Gather inputs/context** — the full dataset the tool should cover, and any notes from prior
   work on the previous version, so the pre-batch grid covers the real use cases.

---

## 1. What EXBYCost is today

A Streamlit techno-economic analysis (TEA) tool that estimates extraction costs from
food-processing byproducts, built on the **BioSTEAM** biorefinery simulation framework.

| Component | File | Role |
|---|---|---|
| Main app | `extraction_tea_tool.py` (~3,550 lines) | Streamlit UI + simulation orchestration |
| Solid–solvent extractor unit | `SolidSolventExtractor.py` | Custom BioSTEAM unit op |
| Proxy finder | `biosteam_proxy_finder.py` | Thermo proxies for unknown chemicals |
| Polymer solubility | `Polymersolubility.py` | Hansen / solubility modeling |
| Biopolymer SMILES | `biopolymer_smiles.py` | ChEBI ontology → SMILES |
| Temperature thresholds | `temperature_thresholds.py` | Degradation / boiling classification |
| Pricing | `pricing.py` | FRED / EIA / BLS price fetchers |
| Mill | `Mill.py` | Custom size-reduction unit |
| Data | `chebi_lite.obo` (53 MB), `solvent_prices.csv` | ChEBI dump + price table |

**Two modes:** *Single Run* (full TEA report) and *Parameter Sweep* (vary N inputs over
ranges, run all combinations). The sweep currently runs in a background `threading.Thread`
(`_sweep_worker`), writes crash-safe JSONL, supports resume after reconnect, and shows live
progress + ETA. Simulations are serialized by a global `_SIM_LOCK` because BioSTEAM uses a
singleton `main_flowsheet`.

**External APIs** (`FRED_API_KEY`, `EIA_API_KEY`, `BLS_API_KEY`) are read from `st.secrets`
with an `os.environ` fallback. **Graphviz** (apt package, see `packages.txt`) is used for
flowsheet diagrams.

---

## 2. The core architectural insight (read this first)

`_run_one_simulation` does **two very different things** back-to-back:

1. **Physical simulation (EXPENSIVE).** `sys.simulate()` (`extraction_tea_tool.py:2153` /
   `:2189`) solves mass/energy balances and sizes equipment. This is the slow BioSTEAM part,
   serialized by `_SIM_LOCK`.
2. **Economic / TEA layer (CHEAP).** `ExtractionTEA(...)` (`:2273`) is constructed **from the
   already-simulated system**; then `tea.solve_price()`, `tea.FCI`, `tea.FOC`, `tea.NPV`, and
   the cost fractions derive everything. This is sub-second arithmetic.

**Parameter classification by the layer it touches:**

| Group | Touches | Pre-batch? |
|---|---|---|
| feedstock, solvent, reactor type, heat utility (discrete) | **Physical** | ✅ grid |
| feedflow, solvflow, ExtractT, extractt, particle size, pressure, evaporator effects, target solids, dryer moisture, recycle | **Physical** | ✅ grid (quantized) |
| **IRR, income tax, depreciation, finance terms, maintenance, insurance, property tax, labor/wage, operating days, startup fractions, elec/feed price, supervision, supplies, admin, WC/FCI, indirect costs … (~25 of ~35 numeric params)** | **Economic only** | ❌ recompute **live** |

> **~25 of the ~35 sweepable numeric parameters never need a re-simulation.** They are TEA
> constructor arguments. Re-instantiating `ExtractionTEA` on a cached, already-simulated
> `System` and calling `solve_price()` is milliseconds.

This is the whole strategy: **pre-batch the ~9 physical dimensions; let the ~25 economic
dimensions float live off the cache.**

---

## 3. Why a full factorial sweep is the wrong target

Even restricted to physical dimensions, the full grid explodes:
`4 feedstocks × ~5 solvents × 3 reactors × ~3 heat utilities × 5 flow × 5 temp × 5 time ×
4 particle × 3 pressure …` → millions of simulations. **Do not attempt an exhaustive grid.**

Instead:

1. **Lazy memoization, not exhaustive pre-batch.** Cache keyed by a hash of the *physical*
   params; compute on miss, store, reuse on hit. Pre-batch only the popular corners (all
   feedstock × solvent × reactor combos at a coarse flow/temp/time grid) to warm the cache —
   a few thousand sims, feasible in Deepnote's 48-hr EDU window.
2. **Quantize continuous physical inputs** to the grid resolution before hashing, so 35.0 °C
   and 35.2 °C hit the same entry. Snap the app's sliders to that grid.
3. **(Optional) interpolation** between grid points for smooth quantities — but **caveat:**
   equipment sizing has discontinuities (adding an evaporator effect, regime switches), so
   interpolated capital can be wrong. Prefer cache-and-snap over interpolation.

### 3.1 Parallelism — the actual speedup ⚡

The sweep is slow because it runs **serially**, guarded by the global `_SIM_LOCK`. That lock
exists for a real reason: **BioSTEAM uses a singleton `bst.main_flowsheet`, so two simulations
cannot run in the same process at once.** Therefore:

- **Threads do not help.** Python threads share one interpreter and one flowsheet — they would
  collide on the singleton (and the GIL blocks CPU-bound work anyway). The existing background
  thread only keeps the UI responsive; it does **not** add throughput.
- **Processes do.** Run the pre-batch with a `concurrent.futures.ProcessPoolExecutor` (or
  `multiprocessing.Pool`) sized to the machine's cores. Each worker is a **separate process with
  its own `main_flowsheet`**, so N cores ≈ **N× throughput**. A grid that was "a couple of days"
  serial becomes a few **hours**. (Keep `_SIM_LOCK` for any in-process safety, but the parallel
  win is across processes, not threads.)
- **Amortize per-worker setup.** Thermo/chemicals init, the 53 MB ChEBI load, and proxy-finding
  are expensive. Do them **once per worker** via a pool `initializer`, then have each worker chew
  through a chunk of combinations — don't pay setup per task.
- **Single writer.** Workers return scalars to the parent; the **parent** writes them to DuckDB
  (one writer, §4). Don't let workers write the DB concurrently.
- **Scale beyond one machine** if needed: shard the grid by range across multiple Deepnote
  notebooks/projects, each writing a parquet shard, then merge into DuckDB. (Belt-and-suspenders
  on top of multiprocessing; usually not necessary once the batch is process-parallel.)
- **The interactive app needs no parallelism** — once results are cached, requests are instant
  lookups, which also fixes the "multiple users slow it down" problem.

---

## 4. Cache design

The mental model from the design discussion: **run every physical permutation once into a
master store, then "recombine" cached results on demand** instead of re-simulating. With ~25-min
jobs, even a few hundred combinations is a one-time batch of a few days of compute — after that
the tool is effectively instant.

```text
key   = sha256( json.dumps(quantized_physical_params, sort_keys=True) + sim_version )
value = physical result needed to rebuild a TEA:
        installed equipment costs, utility duties, stream flows + costs, unit sizing
store = DuckDB table (recommended):
        { key, params_json, physical_json, sim_version, created_at }
```

**Storage choice — DuckDB over a master CSV.** DuckDB is free, file-based, and very fast for the
read-heavy lookups the app does. Its one real limitation — **a single writer at a time** — does
not hurt here, because writes happen only in the **one** batch precompute process, while the app
and all its users are **read-only**. (Keep raw grid output as parquet too if convenient; DuckDB
reads parquet directly.) A plain CSV works but is slower and clumsier to query at scale.

**Rules:**
- **`sim_version` is mandatory in the key.** Bump it on *any* change to the physical model
  code, or you will serve physics computed by stale code. This is the #1 failure mode — and it
  is exactly the "I found an error and had to redo everything" problem, solved cleanly: bump the
  version and only the affected rows recompute.
- Cache **miss** behavior in the app: either (a) run one live sim (acceptable for single-user
  / admin use, but it blocks others via `_SIM_LOCK`), or (b) enqueue a precompute job and show
  "computing…". Prefer (b) for multi-user.
- Cache **hit**: rebuild `ExtractionTEA` with the live economic sliders → render. Instant.
- **Resumable batch:** the precompute writes each row as it finishes, so a crash/disconnect
  resumes from where it stopped — no lost work (preserve the existing JSONL-resume guarantee).

---

## 5. Deepnote target architecture

Deepnote **natively hosts Streamlit** (drop the `.py` into the project's *Files* section and it
auto-deploys; project hardware serves it). So the app ships with minimal code change.

Heavy compute uses the **precompute-offload pattern** (a proven approach for offloading batch
work to Deepnote): an external trigger calls the Deepnote **Execute API**, a runner notebook
computes results and writes them to storage, and the app reads them.

```
┌─────────────────────┐   Execute API    ┌──────────────────────────┐
│ External trigger    │ ───(wake then──▶ │ Deepnote runner notebook │
│ (systemd timer or   │     run)         │  simulate_physical() grid │
│  GitHub Actions —    │                  │  → write cache (parquet)  │
│  NOT native cron)    │                  └────────────┬─────────────┘
└─────────────────────┘                               │ writes
                                                       ▼
┌─────────────────────┐    lookup + live TEA   ┌──────────────────┐
│ Deepnote Streamlit  │ ◀───────────────────── │  cache store      │
│ app (this repo)     │                        │  (parquet/DuckDB) │
└─────────────────────┘                        └──────────────────┘
```

**Hard constraints (from hard-won operational experience):**
- **No Deepnote native cron.** All scheduling goes through an external trigger hitting the
  Execute API: `POST https://api.deepnote.com/v1/projects/<pid>/notebooks/<nb>/execute`.
- **Wake-then-run.** A bare execute on a stopped (idle) machine returns `202` but runs nothing.
  Execute a lightweight keepalive notebook first, wait ~90 s for boot, then execute the work.
- **The Execute API cannot confirm completion** (no status endpoint). **Verify by output
  freshness** (check the cache file/blob timestamp), never trust the `202`.
- Machines shut down on idle / uptime limit / quota (EDU tier: free, 48-hr runtime). Long
  background jobs belong in the runner notebook, not the interactive app.

---

## 6. Phased plan

### Phase 0 — Setup (½ day)
- Complete the **Prerequisites** in §0 first (Deepnote access via GitHub Student Pack; API keys).
- Create the Deepnote project; choose a CPU-strong machine (BioSTEAM is CPU-bound).
- Add secrets as **environment variables** (`FRED_API_KEY`, `EIA_API_KEY`, `BLS_API_KEY`).
- Build a **custom environment** that provides `graphviz` (custom Docker image, or
  `conda install graphviz` + an **Init notebook**). Install `requirements.txt`.
- Upload the 8 `.py` modules + `solvent_prices.csv` + `chebi_lite.obo` to *Files*.

### Phase 1 — Host the Streamlit app (~1 day)
- Drop `extraction_tea_tool.py` into *Files*; verify auto-deploy. **No `st.*` rewrite needed.**
- Replace the `st.secrets` branch with `os.environ` (the fallback already exists).
- Turn **"Run on save" off** so viewers don't see mid-edit states.
- **Milestone:** Single-Run mode live.

### Phase 2 — Split physical vs economic (the enabling refactor, ~1 day) ⭐
- Refactor `_run_one_simulation` into:
  - `simulate_physical(physical_params) -> physical_result` (cacheable, expensive)
  - `cost_it(physical_result, economic_params) -> scalars` (cheap, live)
- **Prove the load-bearing assumption first:** confirm `ExtractionTEA` can be rebuilt on a
  retained, already-simulated `System` and re-solve `solve_price()` without re-simulating.
- Pre-process `chebi_lite.obo` once into parquet/pickle to cut per-boot load cost.

### Phase 3 — Cache + pre-batch runner (~2–3 days)
- Implement the cache (key/quantization helper + **DuckDB** store from §4).
- Write a **runner notebook** that walks the physical grid via `simulate_physical` using a
  **`ProcessPoolExecutor`** (one worker per core, per-worker `initializer` for thermo/ChEBI
  setup — see §3.1), with the parent writing rows to DuckDB. This is the one-time "master" batch
  — make it resumable so an interruption never restarts from zero.
- Wire the app: lookup by quantized key → `cost_it` with live economic sliders → render.

### Phase 4 — External trigger + verification (~1 day)
- Build a small trigger script (wake-then-run) with an EXBYCost case. Schedule via a systemd
  timer or GitHub Actions. **Not native cron.**
- Add output-freshness verification (e.g. a command that exits 0 only when fresh output
  exists) so a `202` is never mistaken for success.

### Phase 5 — Hardening (~1 day)
- Regression-test `simulate_physical`/`cost_it` against the current monolith on known cases.
- Size machine + inactivity timeout for expected concurrency.
- Scrub inline secrets (Deepnote app permissions expose *all* file/block content).

**Total: ~6–7 working days.** Phase 2 is load-bearing and pays off even if Deepnote is dropped.

---

## 7. Risks & open questions
- **Sim version discipline.** Forgetting to bump `sim_version` serves stale physics. Automate
  it (e.g., hash the physical-model source files).
- **`_SIM_LOCK` / singleton flowsheet.** Sims can't share a process; parallelize with **separate
  processes** (`ProcessPoolExecutor`), not threads (§3.1). Verify each worker gets a clean
  `main_flowsheet` and results are reproducible vs the serial baseline.
- **Interpolation correctness.** Capital-cost discontinuities make interpolation risky —
  default to cache-and-snap.
- **Native-cron rule** — confirm it still holds for this project (treat it as absolute unless a
  maintainer says otherwise).

## 8. References
- Deepnote data apps: https://deepnote.com/docs/data-apps
- Deepnote input blocks: https://deepnote.com/docs/input-blocks
- Deepnote Streamlit hosting: https://deepnote.com/docs/streamlit
- Deepnote long-running jobs: https://deepnote.com/docs/long-running-jobs
- Deepnote Execute API: https://apidocs.deepnote.com
- BioSTEAM: https://github.com/BioSTEAMDevelopmentGroup/biosteam
