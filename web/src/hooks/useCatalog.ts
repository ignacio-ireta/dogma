import { useQuery } from "@tanstack/react-query";
import { getCatalog } from "../api/endpoints";

export function useCatalog() {
  return useQuery({ queryKey: ["catalog"], queryFn: getCatalog });
}
