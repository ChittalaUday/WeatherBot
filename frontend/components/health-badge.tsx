"use client";

import { useQuery } from "@tanstack/react-query";
import { Wifi, WifiOff } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { getApiUrl } from "@/lib/utils";

/** Backend + model health over TanStack Query; socket state comes from the socket itself. */
export function HealthBadge({ connected }: { connected: boolean }) {
  const { data } = useQuery({
    queryKey: ["health"],
    queryFn: async () => (await fetch(`${getApiUrl()}/api/health`)).json(),
    refetchInterval: 15_000,
  });

  const live = connected && data?.model;
  return (
    <Badge variant={live ? "secondary" : "outline"} className="gap-1.5 text-[11px] font-normal">
      {live ? (
        <Wifi className="h-3 w-3 text-emerald-500" />
      ) : (
        <WifiOff className="h-3 w-3 animate-pulse text-amber-500" />
      )}
      {live ? "model live" : connected ? "connecting" : "offline"}
    </Badge>
  );
}
