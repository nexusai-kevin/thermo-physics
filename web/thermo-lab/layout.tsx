import { Fraunces, Inter, JetBrains_Mono } from "next/font/google";

const fraunces = Fraunces({
  subsets: ["latin"],
  weight: ["300", "400", "500"],
  style: ["normal", "italic"],
  variable: "--font-display",
  display: "swap",
});

const inter = Inter({
  subsets: ["latin"],
  weight: ["300", "400", "500", "600"],
  variable: "--font-sans",
  display: "swap",
});

const jetbrainsMono = JetBrains_Mono({
  subsets: ["latin"],
  weight: ["400", "500"],
  variable: "--font-mono",
  display: "swap",
});

export default function ThermoLabLayout({ children }: { children: React.ReactNode }) {
  return (
    <div
      className={`${fraunces.variable} ${inter.variable} ${jetbrainsMono.variable} min-h-screen bg-[#020308] text-[#faf9f5] font-[family-name:var(--font-sans)]`}
      style={{
        ["--color-bg" as string]: "#020308",
        ["--color-ink" as string]: "#faf9f5",
        ["--color-ink-dim" as string]: "rgba(250, 249, 245, 0.68)",
        ["--color-ink-faint" as string]: "rgba(250, 249, 245, 0.38)",
        ["--color-line" as string]: "rgba(250, 249, 245, 0.10)",
      }}
    >
      {children}
    </div>
  );
}
