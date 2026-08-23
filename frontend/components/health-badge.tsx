"use client";

import { useQuery } from "@tanstack/react-query";
import { Wifi, WifiOff } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { apiUrl } from "@/lib/utils";

/**
 * Is the backend answering, and is a model loaded.
 *
 * With the socket gone there is no connection whose state means anything between turns, so
 * this is the whole health signal: a poll of /api/health. A failed poll is "offline" - which
 * is what a user needs to know, and is more honest than a socket that is open to a process
 * whose model failed to load.
 */
export function HealthBadge() {
  const { data, isError } = useQuery({
    queryKey: ["health"],
    queryFn: async () => {
      const response = await fetch(`${apiUrl()}/api/health`);
      if (!response.ok) throw new Error(String(response.status));
      return response.json();
    },
    refetchInterval: 15_000,
    retry: false,
  });

  const live = !isError && (data?.models ?? []).some((m: { present: boolean }) => m.present);
  return (
    <Badge variant={live ? "secondary" : "outline"} className="gap-1.5 text-[11px] font-normal">
      {live ? (
        <Wifi className="h-3 w-3 text-emerald-500" />
      ) : (
        <WifiOff className="h-3 w-3 animate-pulse text-amber-500" />
      )}
      {live ? "model live" : data ? "no model" : "offline"}
    </Badge>
  );
}
