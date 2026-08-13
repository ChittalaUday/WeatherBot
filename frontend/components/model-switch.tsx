"use client";

import { useQuery } from "@tanstack/react-query";
import { Layers } from "lucide-react";

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8787";

type ModelInfo = { version: string; present: boolean; size_mb: number | null; description: string };

/** Pick which NLU version answers. v1 and v2 are both live; the choice rides on each query. */
export function ModelSwitch({
  value,
  onChange,
}: {
  value: string;
  onChange: (version: string) => void;
}) {
  const { data } = useQuery({
    queryKey: ["models"],
    queryFn: async () => (await fetch(`${API}/api/models`)).json(),
    staleTime: 5 * 60_000,
  });

  const models: ModelInfo[] = (data?.available ?? []).filter((m: ModelInfo) => m.present);
  if (models.length < 2) return null;

  return (
    <div className="flex items-center gap-1 rounded-full border p-0.5" role="radiogroup" aria-label="NLU model">
      <Layers className="ml-1.5 h-3 w-3 text-muted-foreground" />
      {models.map((model) => (
        <button
          key={model.version}
          role="radio"
          aria-checked={value === model.version}
          title={`${model.description}${model.size_mb ? ` · ${model.size_mb} MB` : ""}`}
          onClick={() => onChange(model.version)}
          className={
            "rounded-full px-2.5 py-1 text-[11px] font-medium transition-colors " +
            (value === model.version
              ? "bg-primary text-primary-foreground"
              : "text-muted-foreground hover:text-foreground")
          }
        >
          {model.version}
        </button>
      ))}
    </div>
  );
}
