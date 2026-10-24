import { useEffect } from "react";
import { Hero } from "@/components/landing/Hero";
import { Engine } from "@/components/landing/Engine";
import { OurModel } from "@/components/landing/OurModel";
import { Benchmarks } from "@/components/landing/Benchmarks";
import { Cta } from "@/components/landing/Cta";

/** Landing (Pen frame pOKAY, "Landing v2 · Compute intelligence (A+)"). Numbers: data/frontend/showcase.json. */
export default function Landing() {
  useEffect(() => {
    document.title = "Busy Machines";
  }, []);
  return (
    <main className="min-w-[1280px] bg-cream">
      <Hero />
      <Engine />
      <OurModel />
      <Benchmarks />
      <Cta />
    </main>
  );
}
