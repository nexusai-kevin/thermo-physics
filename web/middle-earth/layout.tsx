import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Middle-earth Thermodynamic Routing Engine | Nexus AI",
  description:
    "Interactive Fellowship path optimization across Middle-earth modeled as a continuous-discrete potential energy surface with Kramers barrier hopping and combinatorial path action relaxation.",
  openGraph: {
    title: "Middle-earth Thermodynamic Routing Engine & Esoteric Systems Atlas",
    description:
      "Interactive Middle-earth path optimization across 18 canonical waypoints with real-time Kramers barrier hopping, Sauron vigilance fields, and Langevin path annealing.",
  },
};

export default function MiddleEarthLayout({ children }: { children: React.ReactNode }) {
  return children;
}
