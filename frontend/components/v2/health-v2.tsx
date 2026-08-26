"use client";

import { useQuery } from "@tanstack/react-query";
import { Wifi, WifiOff } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { v2Url } from "@/lib/v2/api";
import type { Parser } from "@/lib/v2/types";

type Model = { version: Parser; present: boolean };

/**
 * Is v2 answering, and which parsers are loaded.
 *
 * Polls v2's own /api/health, so it says nothing about the v1 backend - the two are separate
 * processes and a badge that conflated them would be worse than no badge.
 */
export function HealthV2({ onParsers }: { onParsers?: (parsers: Parser[]) => void }) {
  const { data, isError } = useQuery({
    queryKey: ["v2-health"],
    queryFn: async () => {
      const response = await fetch(`${v2Url()}/api/health`);
      if (!response.ok) throw new Error(String(response.status));
      const body = (await response.json()) as { models: Model[] };
      onParsers?.(body.models.filter((m) => m.present).map((m) => m.version));
      return body;
    },
    refetchInterval: 20_000,
    retry: false,
  });

  const live = (data?.models ?? []).filter((model) => model.present);
  const label = isError || !data ? "v2 offline" : live.map((m) => m.version).join(" + ");

  return (
    <Badge
      variant={live.length ? "secondary" : "outline"}
      className="gap-1.5 text-[11px] font-normal"
      title={isError ? `No answer from ${v2Url()}` : `${v2Url()} · parsers: ${label}`}
    >
      {live.length ? (
        <Wifi className="h-3 w-3 text-emerald-500" />
      ) : (
        <WifiOff className="h-3 w-3 animate-pulse text-amber-500" />
      )}
      {label}
    </Badge>
  );
}
