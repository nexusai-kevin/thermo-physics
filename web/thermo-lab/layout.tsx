import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Thermo Lab — Heat is the computer",
  description:
    "Thermo Lab — thermodynamic sampling research from Nexus AI Solutions: 16 benchmark experiments, an offline Potts codon sampler for academic biofoundries, a Potts dispatcher, and an Ising microgrid solver. MIT open source.",
  alternates: {
    canonical: "/thermo-lab",
  },
  openGraph: {
    title: "Thermo Lab — Heat is the computer",
    description:
      "Thermo Lab — thermodynamic sampling research from Nexus AI Solutions: 16 benchmark experiments, an offline Potts codon sampler for academic biofoundries, a Potts dispatcher, and an Ising microgrid solver. MIT open source.",
    url: "/thermo-lab",
    type: "website",
  },
  twitter: {
    card: "summary_large_image",
    title: "Thermo Lab — Heat is the computer",
    description:
      "Thermo Lab — thermodynamic sampling research from Nexus AI Solutions: 16 benchmark experiments, an offline Potts codon sampler for academic biofoundries, a Potts dispatcher, and an Ising microgrid solver. MIT open source.",
  },
};

export default function ThermoLabLayout({ children }: { children: React.ReactNode }) {
  return (
    <div className="thermo-lab-root" data-no-mobile-cta="">
      {children}
    </div>
  );
}
