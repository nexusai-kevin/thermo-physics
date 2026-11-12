# Editorial Redesign Gameplan (Academic Research Bench Style)

This specification translates the typography, surface tokens, scrollytelling layout, and micro-interaction patterns into an actionable execution plan for the thermodynamic pages in `C:\Users\mcder\Desktop\nexus-site-temp`.

---

## 1. Editorial Research Design DNA

1. **Three-Font Editorial Typography Stack**:
   - **Display / Headlines (`--font-display`)**: **`Fraunces`** (Google Font, variable optical size `opsz 9..144`, weights `300, 400, 500`, normal & italic). Headlines use `font-normal leading-[0.95] tracking-[-0.03em] text-[clamp(40px,6.5vw,96px)]`, with the second half of the sentence wrapped in an italicized, lightweight accent span:
     ```tsx
     <em className="italic font-light text-[color:var(--accent)]">into place.</em>
     ```
   - **Body (`--font-sans`)**: **`Inter`** (`font-light`, weights `300, 400, 500`), set at `text-[clamp(15px,1.4vw,18px)] leading-relaxed text-[var(--color-ink-dim)]`.
   - **Telemetry / Kickers (`--font-mono`)**: **`JetBrains Mono`** (weights `400, 500`), used for numbered section indices (`01 / 04 · /codon`), code-comment eyebrows (`// offline potts codon sampler · mit`), and uppercase pill CTAs (`text-[11px] uppercase tracking-[0.15em]`).

2. **Obsidian & Warm-Ivory Surface Tokens + Film Grain**:
   - Deep obsidian background (`#020308`) instead of flat gray.
   - Warm ivory primary ink (`#faf9f5`), muted secondary ink (`rgba(250,249,245,0.68)`), faint ink (`rgba(250,249,245,0.38)`), and 1px hairline borders (`rgba(250,249,245,0.10)`).
   - **Single Room Accent (`--accent`) per vertical**:
     - `/codon`: `#5eead4` (bioluminescent teal)
     - `/dispatcher`: `#f59e0b` (warm amber)
     - `/microgrid`: `#93c5fd` (ice blue)
   - **Fixed SVG `fractalNoise` Film-Grain Overlay**: `fixed inset-0 pointer-events-none z-[100] opacity-[0.025] mix-blend-overlay` with `<feTurbulence type="fractalNoise" baseFrequency="0.9" numOctaves="3" />`.

3. **Asymmetric Sticky Split-Screen Scrollytelling (`lg:grid-cols-[1fr_1.3fr]`)**:
   - **Left Column (Sticky Canvas Frame)**: `sticky top-[14vh] h-[68vh] rounded-2xl overflow-hidden border border-[var(--color-line)] bg-[#05070f]` housing the interactive visualizer (`<CodonRibbon />`, `<DispatcherBoard />`, or `<CampusNight />`) with subtle corner monospace telemetry overlays (`// 2026 · MIT` top-left, Fraunces caption bottom-left, status badge bottom-right).
   - **Right Column (Editorial Cards)**: Scrolling stack of cards (`rounded-2xl border border-[var(--color-line)] bg-[#060811]/80 p-7 md:p-9 opacity-75 hover:opacity-100 transition-opacity duration-500`) covering the benchmark comparison, Hamiltonian terms, and CLI commands.

---

## 2. Step-by-Step Implementation Tasks for the Executing Agent

### Task 1: Load Fonts, CSS Variables & Film-Grain Overlay in `src/app/(thermo)/layout.tsx`
Edit `C:\Users\mcder\Desktop\nexus-site-temp\src\app\(thermo)\layout.tsx`:
1. Import `Fraunces`, `Inter`, and `JetBrains_Mono` from `next/font/google`:
   ```tsx
   import { Fraunces, Inter, JetBrains_Mono } from "next/font/google";

   const fraunces = Fraunces({
     subsets: ["latin"],
     weight: ["300", "400", "500"],
     style: ["normal", "italic"],
     variable: "--font-display",
   });
   const inter = Inter({
     subsets: ["latin"],
     weight: ["300", "400", "500", "600"],
     variable: "--font-sans",
   });
   const jetbrainsMono = JetBrains_Mono({
     subsets: ["latin"],
     weight: ["400", "500"],
     variable: "--font-mono",
   });
   ```
2. Apply font variables and core surface variables on the wrapper `<div>`:
   - `--color-bg: #020308`
   - `--color-ink: #faf9f5`
   - `--color-ink-dim: rgba(250, 249, 245, 0.68)`
   - `--color-ink-faint: rgba(250, 249, 245, 0.38)`
   - `--color-line: rgba(250, 249, 245, 0.10)`
3. Add the subtle fixed SVG `fractalNoise` overlay right inside the layout wrapper (`aria-hidden="true"`).
4. Style the `<header>` and `<nav>` with a minimalist editorial research top bar:
   - Minimal top bar (`border-b border-[var(--color-line)] bg-[#020308]/85 backdrop-blur-md px-[5vw] py-4`).
   - Left brand link in `font-mono text-[11px] uppercase tracking-[0.2em] text-[var(--color-ink-dim)]`.
   - Right links (`GitHub · thermo-physics`, `Contact`) as subtle `font-mono text-[11px] uppercase tracking-[0.15em]` pills.
5. Keep semantic `<header>`, `<nav>`, `<main id="main" tabIndex={-1} data-no-mobile-cta="">`, and `<footer>` intact so `scripts/check-site.mjs` passes all landmark checks.

---

### Task 2: Redesign `src/components/thermo-pages/ThermoChrome.tsx` to Editorial Layout
Update `ThermoChrome.tsx` so `<ThermoHero />`, `<ThermoCompare />`, and `<ThermoAuditForm />` adopt the academic editorial layout:
1. **`<ThermoHero />` Props & Markup**:
   - Accept an optional `indexLabel?: string` (e.g. `"01 / 03 · /codon"`), `accentColor?: string` (default `"#5eead4"`), and `headlineAccent?: string` (optional italicized second part of `h1` or render `h1` in `font-[family-name:var(--font-display)] font-normal leading-[0.96] tracking-[-0.03em] text-[clamp(38px,6vw,88px)] text-[var(--color-ink)]`).
   - **CRITICAL CHECK-SITE CONSTRAINT**: `scripts/check-site.mjs` checks that `<h1>` contains the exact string configured in `VERTICALS` (`"Codons anneal into place."`, `"Watch tasks fall into place."`, `"Four generators. One night."`). Keep the text inside `<h1>` identical in plain text, wrapping the final words in `<em className="italic font-light text-[color:var(--accent)]">...</em>` so `stripTags(h1)` still yields the exact string!
     - `/codon`: `Codons anneal <em className="italic font-light text-[color:var(--accent)]">into place.</em>`
     - `/dispatcher`: `Watch tasks <em className="italic font-light text-[color:var(--accent)]">fall into place.</em>`
     - `/microgrid`: `Four generators. <em className="italic font-light text-[color:var(--accent)]">One night.</em>`
   - Render the eyebrow in `font-mono text-[11px] uppercase tracking-[0.2em] text-[color:var(--accent)]` prefixed with `// `.
   - Render the subhead in `font-light text-[clamp(16px,1.4vw,19px)] leading-relaxed text-[var(--color-ink-dim)] max-w-2xl`.
   - **Asymmetric Split-Screen Body (`px-[5vw] py-16 md:py-24 max-w-[1480px] mx-auto`)**:
     - Use `lg:grid lg:grid-cols-[1.05fr_1.25fr] lg:gap-16 xl:gap-20 items-start`:
       - **Left Column (`lg:sticky lg:top-24`)**: The interactive `children` visual wrapped in a `rounded-2xl border border-[var(--color-line)] bg-[#05070f] p-5 md:p-6 shadow-2xl` container with top-left and bottom-right `font-mono text-[10px] uppercase tracking-[0.18em] text-[var(--color-ink-faint)]` telemetry tags.
       - **Right Column**: The scrolling editorial narrative (`<ThermoCompare />` + the vertical's 4-part technical specification cards + `<ThermoAuditForm />`). To enable this clean 2-column scrollytelling flow, allow `<ThermoHero />` or a new `<ThermoEditorialSplit leftVisual={...} rightContent={...} />` wrapper in `ThermoChrome.tsx` to hold the sticky visual on the left and the cards on the right.

2. **`<ThermoCompare />` Restyle**:
   - Replace the boxy table look with a numbered editorial section (`// 02 · benchmark comparison`, `h2` in `font-[family-name:var(--font-display)] font-normal tracking-[-0.02em] text-[clamp(26px,3.2vw,40px)]`).
   - Render the 2 comparison cards with `rounded-2xl border border-[var(--color-line)] bg-[#060811]/90 p-6 md:p-8 transition-opacity duration-500`. Highlight the thermodynamic card with a subtle `border-[color:var(--accent)]/40` hairline and monospace metric readouts.

3. **`<ThermoAuditForm />` & `<ThermoAlso />` Restyle**:
   - Style inputs and buttons with `rounded-full` or `rounded-xl border border-[var(--color-line)] bg-[#05070f]`, and pill buttons (`rounded-full border border-[var(--color-line)] px-5 py-2.5 font-mono text-[11px] uppercase tracking-[0.15em] hover:border-[var(--color-ink)] transition-colors`).

---

### Task 3: Apply Per-Room Accents & Editorial Cards on `/codon`, `/dispatcher`, and `/microgrid`
1. **`src/app/(thermo)/codon/page.tsx`**:
   - Set `--accent: #5eead4` (teal).
   - Ensure `d_discoideum` (*Dictyostelium discoideum*, ~27% coding GC) is listed in Card 1 (`Supported Host Profiles`), and Card 2 states `Optional Assembly & Synthesis Exclusions (Off by Default)` so it matches biofoundry workflows.
   - Style the 4 Biofoundry Architecture cards as `arbor-card` blocks (`rounded-2xl border border-[var(--color-line)] bg-[#060811]/80 p-6 md:p-8 opacity-80 hover:opacity-100 transition-opacity duration-500`) with `font-mono text-[11px] uppercase tracking-[0.18em] text-[color:var(--accent)]` kickers (`// 01 · host tables`, `// 02 · optional synthesis rules`, `// 03 · potts hamiltonian`, `// 04 · air-gapped cli`).
2. **`src/app/(thermo)/dispatcher/page.tsx`**:
   - Set `--accent: #f59e0b` (warm amber).
   - Style the 4 Router Architecture cards with the same `arbor-card` editorial treatment (`// 01 · multi-objective hamiltonian`, `// 02 · zero prompt exposure`, `// 03 · multi-tier backends`, `// 04 · shadow audit cli`).
3. **`src/app/(thermo)/microgrid/page.tsx`**:
   - Set `--accent: #93c5fd` (ice blue).
   - Style the 4 Station 12 SCUC cards with the same `arbor-card` editorial treatment (`// 01 · quadratic penalty hamiltonian`, `// 02 · 4-tier generation hierarchy`, `// 03 · spin-glass transformation`, `// 04 · hardware realism`).

---

### Task 4: Apply the Same Editorial Palette & Typography to `/thermo-lab`
In `C:\Users\mcder\Desktop\nexus-site-temp\src\app\thermo-lab\page.tsx` (and copy to `C:\Users\mcder\Desktop\thermo-physics\web\thermo-lab\page.tsx`):
1. Import or inherit `--font-display` (`Fraunces`), `--font-sans` (`Inter`), and `--font-mono` (`JetBrains Mono`).
2. Replace `bg-[#0d0e11]` with `bg-[#020308] text-[#faf9f5]` and apply `font-[family-name:var(--font-display)]` to the main hero `<h1>` and section `<h2>` headings, with an italicized teal accent phrase in the hero title.
3. Keep all benchmark metrics (`59.1x`, `1,000x`, `100%`, `r = -0.62`, `r = -0.37`) intact.

---

### Task 5: Strict Guardrails (`scripts/check-site.mjs`) & Verification
Before finishing, run:
```powershell
cd C:\Users\mcder\Desktop\nexus-site-temp
npm run build
npm run check
```
Make sure all 202 checks pass:
- **Forbidden characters in visible text on `(thermo)` pages**:
  - NEVER use em dashes (`—` / `\u2014`) or en dashes (`–` / `\u2013`) in visible text. Use standard hyphens (`-`) or middle dots (`·`).
  - NEVER use exclamation marks (`!`), square brackets (`[` or `]`), or dollar signs (`$`) in visible text.
  - NEVER use banned hype words (`revolutionary`, `game-changing`, `guaranteed`, `unlock`, `supercharge`, etc.) or pricing words (`pricing`, `subscription`, `per month`, `retainer`).
- **Required `needles` in `scripts/check-site.mjs` (lines 520-555)**:
  - Keep the exact `h1` text (`"Codons anneal into place."`, `"Watch tasks fall into place."`, `"Four generators. One night."`) and required substrings (`"CAI"`, `"50-nt"`, `"BsaI"`, `"thermo_physics.codon"`, `"28,742"`, `"thermo_physics.router"`, `"400 MW"`, `"station_12"`, etc.) present in the rendered HTML.
- Once `npm run check` outputs `0 failure(s)`, commit and push `main` in both `C:\Users\mcder\Desktop\nexus-site-temp` and `C:\Users\mcder\Desktop\thermo-physics`.
