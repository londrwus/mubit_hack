import { useEffect } from "react";
import { Hero } from "@/components/landing/Hero";
import { HowItWorks } from "@/components/landing/HowItWorks";
import { Cost } from "@/components/landing/Cost";
import { Engine } from "@/components/landing/Engine";
import { OurModel } from "@/components/landing/OurModel";
import { Benchmarks } from "@/components/landing/Benchmarks";
import { Cta } from "@/components/landing/Cta";

/**
 * Landing v3. One job per screen, nothing repeated between consecutive sections:
 * Hero → I How it works (compute intelligence, no numbers) → II Cost (the AI bill) → III Time (waiting strips, then what
 * it turns into; Engine renders TimeToResult) → IV Model score → V Benchmarks (the one recap) → CTA. Numbers: data/frontend/showcase.json + docs/results_log.md.
 */
export default function Landing() {
  useEffect(() => {
    document.title = "Busy Machines";
  }, []);
  return (
    <main className="min-w-[1280px] bg-cream">
      <Hero />
      <HowItWorks />
      <Cost />
      <Engine />
      <OurModel />
      <Benchmarks />
      <Cta />
    </main>
  );
}
