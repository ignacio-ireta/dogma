import { Route, Routes } from "react-router";
import { AppShell } from "./components/AppShell";
import { RootRedirect } from "./components/RootRedirect";
import { MapPage } from "./pages/MapPage";
import { MockPage } from "./pages/MockPage";
import { NotFoundPage } from "./pages/NotFoundPage";
import { ProblemIndex } from "./pages/ProblemIndex";
import { ProblemPage } from "./pages/ProblemPage";
import { ProgressPage } from "./pages/ProgressPage";
import { ReviewPage } from "./pages/ReviewPage";
import { TodayPage } from "./pages/TodayPage";
import { WelcomePage } from "./pages/WelcomePage";

export function App() {
  return (
    <Routes>
      <Route element={<AppShell />}>
        <Route index element={<RootRedirect />} />
        <Route path="welcome" element={<WelcomePage />} />
        <Route path="today" element={<TodayPage />} />
        <Route path="problem" element={<ProblemIndex />} />
        <Route path="problem/:track/:topic/:problem" element={<ProblemPage />} />
        <Route path="map" element={<MapPage />} />
        <Route path="review" element={<ReviewPage />} />
        <Route path="mock" element={<MockPage />} />
        <Route path="progress" element={<ProgressPage />} />
        <Route path="*" element={<NotFoundPage />} />
      </Route>
    </Routes>
  );
}
