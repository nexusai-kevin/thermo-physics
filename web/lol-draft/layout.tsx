import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "League of Legends 5v5 Thermodynamic Draft Solver | Nexus AI",
  description:
    "Interactive 5v5 LoL pick-and-ban optimizer modeled as an N=10 Categorical Potts Spin Graph. Gibbs MCMC sampling for wombo combos, role constraints, and counter-pick equilibration.",
  openGraph: {
    title: "League of Legends 5v5 Thermodynamic Draft Solver",
    description:
      "Interactive 5v5 LoL pick-and-ban optimizer modeled as an N=10 Categorical Potts Spin Graph with real-time Gibbs MCMC sampling.",
  },
};

export default function LolDraftLayout({ children }: { children: React.ReactNode }) {
  return children;
}
