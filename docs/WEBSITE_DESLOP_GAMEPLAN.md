# Website De-Slop & Vertical Silo Gameplan (Handoff Spec for Next Model)

## Context & Problem Statement
When a biofoundry scientist (Dr. Felix Meier, Australian Genome Foundry) visited `https://getnexusaisolutions.com/codon` and `https://getnexusaisolutions.com/thermo-lab`, he immediately called the site "AI slop / vibe coded" because:
1. **Cross-Contamination of Buyer Personas**:
   - `src/app/(thermo)/layout.tsx` wraps `/codon`, `/dispatcher`, and `/microgrid` in the SMB agency `<SiteHeader />` and `<SiteFooter />` (which link to **Restaurants, Golf Courses, Medical Clinics, E-commerce, Missed-Call Text-Back, and "Own your Super Intelligence"**).
   - `src/components/thermo-pages/ThermoChrome.tsx` renders `<ThermoAlso />` at the bottom of each vertical page, cross-linking `/codon` <-> `/dispatcher` <-> `/microgrid` <-> `/thermo-lab`.
   - On `/thermo-lab` (`src/app/thermo-lab/page.tsx`), the top-left `← Home` button links to `/`, which `next.config.ts` rewrites to `/funnel/index.html` (the NYC contractor missed-call sales funnel).
2. **Robotic Compliance Boilerplate on `/codon`, `/dispatcher`, `/microgrid`**:
   - `src/lib/thermo-claims.ts` defines `SIM_BENCH = "simulated benchmark result, results may vary (not customer results)"`, which is pasted 5–6 times per page (even inside `<dd>` metric cards and meta descriptions), plus `"Paid plans come after a verified audit"` repeated 3 times.
3. **"Arcade / Kid-Story" Copy & Neon/Emoji Styling on `/thermo-lab`**:
   - `src/app/thermo-lab/page.tsx` (and `C:\Users\mcder\Desktop\thermo-physics\web\thermo-lab\page.tsx`) uses titles like *"The Monster Rivalry Game"*, *"4-Color Paint Party"*, *"The 5-Gem Treasure Vault"*, `"THERMO-ARCADE MASTER DIAL"`, emojis on buttons (`🔥`, `❄️`, `🔊`, `📄`), and unmeasured `"0.005s (Extropic TSU)"` / `"28,742x TSU Speedup"` claims.

---

## Repository Locations
- **Live Website Repo (Next.js)**: `C:\Users\mcder\Desktop\nexus-site-temp`
  - **IMPORTANT**: First run `git checkout main && git pull origin main` in `C:\Users\mcder\Desktop\nexus-site-temp` because the `(thermo)` vertical pages (`/codon`, `/dispatcher`, `/microgrid`) live on `origin/main` (commit `db7a8a4`), whereas the working tree is currently checked out on an older branch (`feat/thermo-lab-fable-redesign`).
- **Python & Benchmark Repo**: `C:\Users\mcder\Desktop\thermo-physics`
  - Contains the newly built offline codon package `thermo_physics/codon/` and CLI `scripts/agf_local_codon_benchmark.py`, plus the LLM router package `thermo_physics/router/`.

---

## Step-by-Step Execution Tasks for the Next Model

### Task 1: Sync `C:\Users\mcder\Desktop\nexus-site-temp` to `origin/main`
Run in `C:\Users\mcder\Desktop\nexus-site-temp`:
```powershell
git checkout main
git pull origin main
```

### Task 2: Replace the SMB Agency Header/Footer in `src/app/(thermo)/layout.tsx` with a Clean, Isolated Technical Header/Footer
**File**: `C:\Users\mcder\Desktop\nexus-site-temp\src\app\(thermo)\layout.tsx`
- **Remove** `<SiteHeader />` and `<SiteFooter />` so visitors to `/codon`, `/dispatcher`, and `/microgrid` NEVER see links to Restaurants, Golf, Medical, E-commerce, or the SMB agency homepage (`/`).
- **Replace** with a minimal, monochrome technical `<header>` and `<nav>` and `<footer>` that:
  - Contains `<header>`, `<nav>`, `<main id="main" tabIndex={-1} data-no-mobile-cta="">`, and `<footer>` (so semantic HTML and accessibility checks in `scripts/check-site.mjs` still pass!).
  - Shows only `Nexus Thermodynamic Research` + a link to the GitHub repository (`https://github.com/nexusai-kevin/thermo-physics`) and contact email (`mailto:hello@getnexusaisolutions.com`).
  - Has **zero links** to `/`, `/funnel`, `/digital-resilience`, or other verticals.

### Task 3: Remove Cross-Vertical Links in `src/components/thermo-pages/ThermoChrome.tsx` & `src/content/deck.ts`
**File**: `C:\Users\mcder\Desktop\nexus-site-temp\src\components\thermo-pages\ThermoChrome.tsx`
- Change `ThermoAlso` so it does **not** cross-link `/codon`, `/dispatcher`, and `/microgrid` to each other. Instead, render a single subtle link to the GitHub source repository (`https://github.com/nexusai-kevin/thermo-physics`).
- **File**: `C:\Users\mcder\Desktop\nexus-site-temp\src\content\deck.ts`
  - Remove the `id: "thermo"` group from `deckGroups` so the SMB agency site's hamburger menu doesn't dump restaurant/contractor visitors into `/codon` or `/dispatcher`.

### Task 4: De-Slop `/codon` (`src/app/(thermo)/codon/page.tsx`) and Add the Offline CLI Specification
**File**: `C:\Users\mcder\Desktop\nexus-site-temp\src\app\(thermo)\codon\page.tsx`
1. **Color Palette & Typography**:
   - Replace the glowing matrix-green-on-black (`bg-[#06110c] text-[#e7fff3]` and `#39f0a0`) with a clean, flat, high-contrast academic/documentation palette: neutral zinc (`bg-[#0d0e11] text-zinc-100`, borders `border-zinc-800`, muted text `text-zinc-400`, monospace tables).
2. **Copy Cleanup**:
   - Change the eyebrow from `"Free design-partner trial"` to `"thermo_physics.codon · Offline Potts Codon Sampler (MIT)"`.
   - Remove the repeated `"Paid plans come after a verified audit"` sentences. State clearly: *"Open-source (MIT) Python package. Free permanently for academic labs and university biofoundries."*
   - Keep the required benchmark disclosure once cleanly rather than pasting `SIM_BENCH` under every single number.
3. **Add the Real Biofoundry Capabilities Section (`thermo_physics/codon`)**:
   - Document the 4 things biofoundry engineers care about:
     1. **Supported Host Profiles**: `s_cerevisiae`, `e_coli`, `b_subtilis`, `c_griseus` (CHO), `h_sapiens`.
     2. **Hard Assembly & Synthesis Exclusions**: Type IIS Golden Gate / MoClo (`BsaI`, `BsmBI`/`Esp3I`, `BbsI`/`BpiI`, `SapI`/`LguI`, `AarI`) checked on both strands including codon junctions, plus homopolymer runs ($\ge 5\text{ nt}$).
     3. **Decomposed Potts Hamiltonian $H(s)$**: Host CAI (`-mean ln w_k`), rare-codon doublets, Turner 2004 5′ translation initiation secondary structure ($\Delta G_{5'}$ over nt 1..54), internal stem-loops, and 50-nt sliding-window GC homeostasis.
     4. **100% Local / Air-Gapped Retrospective CLI**:
        ```bash
        python scripts/agf_local_codon_benchmark.py correlate \
          --csv constructs.csv --host s_cerevisiae --enzymes BsaI,BsmBI,BbsI,SapI \
          --summary-out summary_correlations_only.json
        ```
        Explain that `--summary-out` outputs only sample counts, Spearman $\rho$, Pearson $r$, $p$-values, and ROC-AUC with **zero construct sequences or IDs**.

### Task 5: De-Slop `/thermo-lab` (`src/app/thermo-lab/page.tsx` and `thermo-physics/web/thermo-lab/page.tsx`)
1. **Remove the `← Home` link** (`href="/"`) in the top bar so researchers are never sent to `/funnel`. Replace it with a link to the GitHub repository (`https://github.com/nexusai-kevin/thermo-physics`).
2. **Remove the embedded `DispatcherRoiBenchmark` sales widget** from `/thermo-lab` (keep `/thermo-lab` strictly as the 16-station statistical physics benchmark suite).
3. **Replace Arcade / Kid Titles and Emojis in `STATIONS`**:
   - Station 1: `"The 1D Magnet Line"` -> `"1D Ferromagnetic Ising Chain"`
   - Station 2: `"Giant 2D Checkerboard"` -> `"2D Square-Lattice Ising Model (Onsager Criticality)"`
   - Station 3: `"The Monster Rivalry Game"` -> `"Antiferromagnetic Max-Cut on Erdős–Rényi Graphs"`
   - Station 4: `"4-Color Paint Party"` -> `"2D Categorical Potts Model (q = 4)"`
   - Station 5: `"The Magic Pixel Eraser"` -> `"Clamped Ising Denoising Markov Random Field"`
   - Station 6: `"The Half-Erased Memory Vault"` -> `"Bipartite Restricted Boltzmann Machine (RBM)"`
   - Station 7: `"The 5-Gem Treasure Vault"` -> `"Cardinality-Constrained Markowitz QUBO"`
   - Station 8: `"The Analog Skatepark (Two Valley Wells)"` -> `"Coupled Quartic Double-Well Langevin Chain"`
   - Station 9: `"The Tangled Delivery Route"` -> `"10-City Traveling Salesperson Permutation QUBO"`
   - Station 10: `"The Frozen Glass Maze"` -> `"Sherrington–Kirkpatrick Frustrated Spin Glass"`
   - Station 11: `"The RNA Origami Dance"` -> `"RNA Secondary-Structure Boltzmann Ensemble"`
   - Station 12: `"The Industrial Power Grid"` -> `"Unit Commitment Dispatch QUBO (4-Gen × 6-Hour)"`
   - Rewrite the `kidStory` strings into concise, plain-English technical descriptions (no monsters, confetti, treasure chests, or exclamation marks).
   - Remove unmeasured `"/ 0.005s (Extropic TSU)"` and `"28,742× TSU Speedup"` strings from `runtime` and `keyMetric` in Stations 12–16; report only measured JAX/CPU runtimes and label hardware comparisons as analytical projections.
   - Strip emojis (`🔥`, `❄️`, `⚡`, `🔊`, `📄`) from the top bar and preset buttons.

### Task 6: Update `scripts/check-site.mjs` and Verify Build
**File**: `C:\Users\mcder\Desktop\nexus-site-temp\scripts\check-site.mjs`
- Update the `needles` list in section `18. Thermo vertical landing pages` (around line 505) if you remove `"Paid plans come after a verified audit"` from `/codon`, `/dispatcher`, and `/microgrid`.
- Run:
  ```powershell
  npm run build
  npm run check
  ```
  Ensure 100% of checks in `scripts/check-site.mjs` pass before committing and pushing to `origin/main`.
