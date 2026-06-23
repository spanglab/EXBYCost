# AGENTS.md — Instructions for AI Agents Working on EXBYCost

This file orients automated coding agents (Claude Code, Codex, etc.) working in this
repository. Read it fully before making changes. Humans: see `README.md` and
`DEVELOPMENT_PLAN.md`.

---

## 1. What this repo is

**EXBYCost** is a Streamlit techno-economic analysis (TEA) tool that estimates extraction
costs from food-processing byproducts, built on the **BioSTEAM** biorefinery simulation
framework. The active migration goal is documented in **`DEVELOPMENT_PLAN.md`** (Deepnote
hosting + a pre-batched physical-simulation cache with a live economic layer).

## 2. Repository map

| Path | What it is | Touch with care? |
|---|---|---|
| `extraction_tea_tool.py` | Main Streamlit app + simulation orchestration (~3,550 lines) | ⚠️ Yes — core |
| `SolidSolventExtractor.py` | Custom BioSTEAM solid–solvent extractor unit | ⚠️ Physical model |
| `Mill.py` | Custom size-reduction unit | ⚠️ Physical model |
| `biosteam_proxy_finder.py` | Thermo proxy assignment for unknown chemicals | ⚠️ Physical model |
| `Polymersolubility.py` | Hansen / solubility modeling | ⚠️ Physical model |
| `biopolymer_smiles.py` | ChEBI ontology → SMILES resolution | |
| `temperature_thresholds.py` | Degradation / boiling classification | |
| `pricing.py` | FRED / EIA / BLS price fetchers | |
| `chebi_lite.obo` (53 MB) | ChEBI ontology dump | 🚫 Do not edit; do not regenerate casually |
| `solvent_prices.csv` | Solvent base prices + FRED series | |
| `requirements.txt` | Python deps | |
| `packages.txt` | apt packages (`graphviz`) for the host | |
| `.streamlit/` | Streamlit config + (gitignored) secrets | 🚫 Never commit secrets |

## 3. Critical architecture facts (do not relearn these the hard way)

1. **Two computation layers live inside `_run_one_simulation`:**
   - **Physical (expensive):** `sys.simulate()` at `extraction_tea_tool.py:2153` / `:2189`.
   - **Economic (cheap):** `ExtractionTEA(...)` at `:2273`, built *from the simulated system*;
     `tea.solve_price()`, `tea.FCI`, `tea.FOC`, `tea.NPV`, cost fractions.
   - **Implication:** ~25 of ~35 numeric sweep params are economic-only and require **no
     re-simulation**. Keep them out of any pre-batch grid. See `DEVELOPMENT_PLAN.md` §2.
2. **BioSTEAM uses a singleton `main_flowsheet`.** Simulations are **not** thread/parallel-safe
   in-process. The code serializes them with the global `_SIM_LOCK`. Never run two
   `sys.simulate()` calls concurrently in one process. To speed up the pre-batch, parallelize
   with **separate processes** (`concurrent.futures.ProcessPoolExecutor`, one worker per core,
   per-worker setup via an `initializer`) — **not threads** (threads share the flowsheet and the
   GIL, so they add no throughput). See `DEVELOPMENT_PLAN.md` §3.1.
3. **The parameter sweep** runs in a background `threading.Thread` (`_sweep_worker`), persists
   crash-safe JSONL, and supports resume. Preserve these guarantees if you refactor it.
4. **Secrets:** `FRED_API_KEY`, `EIA_API_KEY`, `BLS_API_KEY` come from `st.secrets` with an
   `os.environ` fallback. Never hardcode keys. Never commit `.streamlit/secrets*`.
5. **Determinism / caching:** if you add a physical-result cache, the cache key **must** include
   a `sim_version` that bumps on any physical-model code change (see plan §4). Serving cached
   physics from stale code is the top failure mode. The cache store is **DuckDB** (single writer
   = the batch precompute only; the app and its users are **read-only**). Never write to the
   cache from the interactive app.

## 4. Environment & running

```bash
# Local
pip install -r requirements.txt          # Python 3.10+ (3.11 in devcontainer)
# graphviz must be installed at the OS level (see packages.txt) for flowsheet diagrams
streamlit run extraction_tea_tool.py
```

- Target platform for migration is **Deepnote** (native Streamlit hosting). Heavy compute uses
  the precompute-offload pattern. See `DEVELOPMENT_PLAN.md` §5.
- **Deepnote scheduling rule: NO native cron.** All scheduling goes through an external trigger
  hitting the Execute API, with wake-then-run, and output-freshness verification (the API does
  not confirm completion). See `DEVELOPMENT_PLAN.md` §5 for the full pattern.

## 5. How to make changes

- **Match the surrounding style.** This is a single-file-heavy Streamlit app with helper
  functions prefixed `_`. Keep comment density and naming consistent with neighbors.
- **Do not split or rename `extraction_tea_tool.py` wholesale** without a stated plan; many
  helpers and `st.*` calls are interleaved. The sanctioned refactor is the physical/economic
  split in `DEVELOPMENT_PLAN.md` §6 Phase 2 — do that surgically.
- **Physical-model edits** (`SolidSolventExtractor.py`, `Mill.py`, proxy finder, solubility)
  change TEA outputs. Call it out explicitly in the PR and bump `sim_version` if a cache exists.
- **Keep the two modes working:** Single Run and Parameter Sweep. Don't break resume/JSONL.

## 6. Verification before you call it done

There is no formal test suite yet. At minimum:

```bash
python -c "import ast; ast.parse(open('extraction_tea_tool.py').read())"   # syntax
streamlit run extraction_tea_tool.py                                        # boots clean
```

Then manually: run **Single Run** with defaults and confirm a full TEA report renders; run a
small **Parameter Sweep** (2–3 points on one param) and confirm progress, results, and CSV
download. If you added the physical/economic split, prove that changing an economic-only param
(e.g., IRR) updates outputs **without** re-simulating. If you add tests, document how to run
them here.

## 7. Git & commit conventions

Good git hygiene is required — the owner is not a git expert, so the history must stay clean,
readable, and safe to revert.

**Committing**
- **Commit only when work is in a known-good state** (it imports, the app boots, the change is
  coherent). Never commit half-finished or broken code to `main`.
- **One logical change per commit.** Don't bundle an unrelated refactor, a bugfix, and a docs
  edit into one commit. Small, focused commits are easy to review and revert.
- **Write clear messages.** First line = imperative summary ≤ ~72 chars with a type prefix
  (`feat:`, `fix:`, `docs:`, `refactor:`, `chore:`, `test:`). Add a blank line, then a body
  explaining **what changed and why** (not how). Example:
  ```
  feat: cache physical sim results in DuckDB

  Pre-batch the physical grid once and look results up at request time so
  economic-only inputs (IRR, tax) recompute instantly without re-simulating.
  ```
- **Review before you commit.** Run `git status` and `git diff --staged` and confirm every
  staged file belongs. Stage specific paths (`git add path/to/file`) rather than `git add -A`.
- **Never commit:** API keys, `.streamlit/secrets*`, `.env*`, `.DS_Store`, `__pycache__/`,
  `*.pyc`, scratch/output dirs, or large regenerated data. Add them to `.gitignore` instead.
- **Don't rewrite published history.** No `push --force` / rebase on `main` once others may have
  pulled. Fix mistakes with a new commit (or `git revert`).
- **Pull before you push** (`git pull --rebase`) to avoid noisy merge commits.

**Branching & PRs**
- **Default branch:** `main`. Keep `main` always working.
- **Branch first** for any non-trivial change: `git switch -c type/short-description`
  (e.g. `feat/duckdb-cache`, `fix/sweep-resume`). Do the work there, then open a Pull Request.
- A PR should describe **what** changed, **why**, and **how it was verified** (see §6). Reference
  files as `path:line` where useful. Keep PRs small enough to review in one sitting.
- External contributors (no write access): **fork** the repo, push the branch to the fork, and
  open a PR from the fork against `main`.
- Use your own configured git identity; do not hardcode anyone's email in repo files.

## 8. Hard "don'ts"
- 🚫 Do not commit API keys or `.streamlit/secrets*`.
- 🚫 Do not run concurrent `sys.simulate()` in one process.
- 🚫 Do not introduce Deepnote native cron for scheduling.
- 🚫 Do not regenerate or hand-edit `chebi_lite.obo`.
- 🚫 Do not serve a physical-result cache without a `sim_version` guard.
