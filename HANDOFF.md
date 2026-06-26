# HANDOFF — where we left off

Rolling session log for the Deepnote migration. Newest entry on top. For the full
plan and phase breakdown see [`DEVELOPMENT_PLAN.md`](DEVELOPMENT_PLAN.md); for agent
guardrails see [`AGENTS.md`](AGENTS.md).

---

## Session — 2026-06-25

### Done today
- **Salvaged post-outage work.** Confirmed nothing was lost: the migration docs
  (`AGENTS.md`, `DEVELOPMENT_PLAN.md`) were already committed as `e88ad6d` and pushed
  to the fork. Working tree was otherwise clean (only local tool artifacts).
- **Opened PR #1** → https://github.com/spanglab/EXBYCost/pull/1
  (`dMac716:docs/deepnote-migration-plan` → `spanglab:main`). Docs-only.
- **Added `.gitignore`** so local tool artifacts (`.DS_Store`, `.firecrawl/`,
  `.gstack/`) and Streamlit secrets stay out of the repo.

### Pick up here tomorrow
The plan is written; **execution starts at Phase 0 → Phase 2**. Suggested order:

1. **Phase 0 (Setup, ½ day)** — `DEVELOPMENT_PLAN.md` §6. Get Deepnote access (GitHub
   Student Pack), create the project on a CPU-strong machine, add the three API keys as
   **env vars**, build a custom env that provides `graphviz`, and upload the 8 `.py`
   modules + `solvent_prices.csv` + `chebi_lite.obo`.
2. **Phase 1 (Host Streamlit, ~1 day)** — drop `extraction_tea_tool.py` in, verify
   auto-deploy, swap the `st.secrets` branch for `os.environ`, turn "Run on save" off.
   Milestone: Single-Run mode live.
3. **Phase 2 ⭐ (the load-bearing refactor, ~1 day)** — split `_run_one_simulation` into
   `simulate_physical(physical_params)` and `cost_it(physical_result, economic_params)`.
   **Prove the key assumption FIRST** before building anything on top of it: confirm
   `ExtractionTEA` can be rebuilt on a retained, already-simulated `System` and re-solve
   `solve_price()` **without re-simulating**. If that doesn't hold, the cache design in
   §3–§4 needs rethinking — so validate it on day one.

### Watch out for (from §7 risks)
- **Singleton flowsheet / `_SIM_LOCK`:** sims can't share a process — parallelize with
  separate **processes** (`ProcessPoolExecutor`), never threads.
- **`sim_version` discipline:** stale physics if not bumped; plan to hash the
  physical-model source files.
- **Native-cron rule:** treat as absolute unless a maintainer says otherwise.

### Open / waiting
- PR #1 awaiting maintainer review — may need scope/wording adjustments.
