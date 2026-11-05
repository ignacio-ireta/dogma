import { apiFetch } from "./client";
import type { components } from "./schema";

type Schemas = components["schemas"];
export type CoverageOut = Schemas["CoverageOut"];
export type CoverageRowOut = Schemas["CoverageRowOut"];
export type Tier = CoverageRowOut["tier"];
export type ProgressOut = Schemas["ProgressOut"];

export function getCoverage(tier?: Tier, track?: string): Promise<CoverageOut> {
  const query = new URLSearchParams();
  if (tier) query.set("tier", tier);
  if (track) query.set("track", track);
  const suffix = query.size ? `?${query}` : "";
  return apiFetch<CoverageOut>(`/coverage${suffix}`);
}

export function getProgress(week?: number): Promise<ProgressOut> {
  return apiFetch<ProgressOut>(week === undefined ? "/progress" : `/progress?week=${week}`);
}
