import { createRoot } from "react-dom/client";
import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import "./index.css";
import Landing from "@/pages/Landing";
import GetStarted from "@/pages/GetStarted";
import Compare from "@/pages/Compare";
import { AppShell } from "@/components/platform/AppShell";

// No StrictMode: its double-run effects would open two replay sessions on the backend per visit.
createRoot(document.getElementById("root")!).render(
  <BrowserRouter>
    <Routes>
      <Route path="/" element={<Landing />} />
      <Route path="/app" element={<AppShell />}>
        <Route index element={<GetStarted />} />
        <Route path="compare" element={<Compare />} />
      </Route>
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  </BrowserRouter>,
);
