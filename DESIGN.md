# Design System & Guidelines (Nexus & Thermodynamic Research)

This document encodes the visual identity, typography hierarchy, surface tokens, and anti-slop rules enforced across all pages in accordance with `impeccable`, `frontend-design`, and `design-taste-frontend`.

---

## 1. Aesthetic Direction: Editorial Scientific Research

Inspired by leading scientific instruments, *Nature Methods*, and *Quanta Magazine*. The site reads as an authoritative, high-craft scientific instrument and academic publication—never as a generic template or promotional AI-SaaS landing page.

### Core Principles
- **Restraint Over Noise**: Spend boldness in one place (e.g. the live interactive ribbon or network graph). Keep everything around it quiet, disciplined, and razor-sharp.
- **Asymmetric Split-Screen Framing**: Sticky visualizer canvas on the left (`lg:sticky lg:top-24`), paced editorial narrative and benchmark evidence scrolling on the right (`lg:grid-cols-[1.1fr_1.3fr]`).
- **Tactile Materiality**: Subtle fixed SVG film grain (`feTurbulence fractalNoise` at 2.5% opacity) eliminates sterile digital flatness.

---

## 2. Color Palette & Surface Tokens

### Obsidian & Warm-Ivory Core
- **Background (`--color-bg`)**: `#020308` (deep obsidian; never washed-out slate or gray)
- **Primary Ink (`--color-ink`)**: `#faf9f5` (warm ivory text; never harsh #ffffff)
- **Secondary Ink (`--color-ink-dim`)**: `rgba(250, 249, 245, 0.68)`
- **Faint Ink (`--color-ink-faint`)**: `rgba(250, 249, 245, 0.38)`
- **Hairlines & Borders (`--color-line`)**: `rgba(250, 249, 245, 0.10)`
- **Elevated Canvas Frame**: `#05070f`
- **Elevated Editorial Card**: `#060811` with 85–90% opacity and subtle backdrop blur

### Dedicated Per-Vertical Accents (`--accent`)
- **`/codon`**: `#5eead4` (bioluminescent teal)
- **`/dispatcher`**: `#f59e0b` (warm amber)
- **`/microgrid`**: `#93c5fd` (ice blue)
- **`/thermo-lab`**: Cryogenic teal (`#5eead4`) & emerald (`#34d399`)

---

## 3. Typography Stack

Three typefaces with clearly defined, non-overlapping roles:

1. **Display / Headlines (`--font-display: Fraunces`)**
   - Google Font, variable optical size `opsz 9..144`, weights 300, 400, 500.
   - Used for main `<h1>` and section `<h2>` headings.
   - Optical scale: `text-[clamp(36px,5.8vw,80px)] font-normal leading-[0.96] tracking-[-0.03em]`.
   - Subtle italic accent spans allowed on closing clauses (`<em className="italic font-light text-[color:var(--accent)]">...</em>`).

2. **Body & Prose (`--font-sans: Inter`)**
   - Weights 300, 400, 500.
   - Clear line lengths `< 75` characters, `leading-relaxed`.
   - Set in `--color-ink-dim` (`rgba(250,249,245,0.68)`).

3. **Telemetry & Kicker Labels (`--font-mono: JetBrains Mono`)**
   - Weights 400, 500.
   - Code kickers (`// 01 · host tables`), section indices (`01 / 03`), status badges, and CLI snippets.
   - Pill CTAs: `font-mono text-[11px] uppercase tracking-[0.15em] rounded-full`.

---

## 4. Strict Anti-Slop Rules (Banned AI Tropes)

- **NO Purple / Blue Gradient Text**: Never use `bg-gradient-to-r from-cyan to-purple` or rainbow gradient fills.
- **NO Generic SaaS-Card Grids**: Do not chop pages into dozens of identical rounded boxes with heavy drop shadows.
- **NO Marketing Fluff**: Zero banned hype words (`revolutionary`, `game-changing`, `guaranteed`, `unlock`, `supercharge`). State empirical numbers plainly.
- **NO Em/En Dashes in Visible Copy**: Use standard hyphens (`-`) or middle dots (`·`).
- **NO Exclamation Points (`!`) or Square Brackets (`[` or `]`)** in visible customer copy.
- **NO Dollar Signs or Pricing Leakages** on research vertical pages.
