import { Link, useNavigate } from "react-router-dom";
import { Columns2, Plus } from "lucide-react";
import { PageHeader } from "@/components/platform/AppShell";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { KpiCards } from "@/components/platform/overview/KpiCards";
import { MachinesHeatmap } from "@/components/platform/overview/MachinesHeatmap";
import { BillBreakdown } from "@/components/platform/overview/BillBreakdown";
import { QualityChart } from "@/components/platform/overview/QualityChart";

/** Overview dashboard (Pen frame bxWh3). Headline numbers: R78 / R82 / R90; grid and curves: stored race 4ca4ed27. */
export default function Overview() {
  const navigate = useNavigate();
  return (
    <>
      <PageHeader
        eyebrow="Workspace"
        title="Overview"
        subtitle="Logged sweeps, from stored race data. Hover any number for its source."
        right={
          <div className="flex items-center gap-2">
            <Link
              to="/app/compare"
              className="inline-flex h-[38px] items-center gap-1.5 rounded-[9px] bg-white px-4 text-[13px] font-semibold text-ink shadow-[0_1px_2px_#1A2F7A0D] outline outline-1 -outline-offset-1 outline-app-line transition-shadow hover:shadow-[0_3px_10px_#1A2F7A18]"
            >
              <Columns2 size={15} /> Watch the replay
            </Link>
            <Button onClick={() => navigate("/app")}>
              <Plus size={15} /> New sweep
            </Button>
          </div>
        }
      />
      <div className="flex items-center gap-2 text-[12px] text-app-slate">
        <Badge variant="blue" size="tag">
          Game-playing sweep · 30 machines · 15 min
        </Badge>
        <Badge variant="grey" size="tag">
          Standard agent and Busy Machines both use Opus 5.5
        </Badge>
      </div>
      <KpiCards />
      <MachinesHeatmap />
      <div className="grid grid-cols-2 gap-4">
        <BillBreakdown />
        <QualityChart />
      </div>
    </>
  );
}
