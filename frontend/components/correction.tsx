"use client";

import { useQuery } from "@tanstack/react-query";
import { Check, X } from "lucide-react";
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { useFeedback } from "@/lib/use-feedback";
import { cn } from "@/lib/utils";

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8787";

/**
 * What a thumbs-down actually needs to be worth keeping.
 *
 * "This was wrong" is triage; "this should have been RAIN in Guntur tomorrow" is a training
 * row. The form is pre-filled with what the model said, so a correction is usually one tap
 * on the label that was wrong.
 */
export function Correction({
  turnId,
  model,
  variables,
  locations,
  times,
  onDone,
}: {
  turnId: number;
  model: string;
  variables: string[];
  locations: string[];
  times: string[];
  onDone: () => void;
}) {
  const feedback = useFeedback();
  const { data } = useQuery({
    queryKey: ["labels"],
    queryFn: async () => (await fetch(`${API}/api/labels`)).json(),
    staleTime: 10 * 60_000,
  });

  // Model 1 predicts a set of variables, never a single intent, so this is always multi-select
  const options: string[] = data?.variables ?? [];
  const [picked, setPicked] = useState<string[]>(variables);
  const [place, setPlace] = useState(locations.join(", "));
  const [when, setWhen] = useState(times.join(", "));
  const [note, setNote] = useState("");

  const toggle = (label: string) =>
    setPicked((current) =>
      current.includes(label)
        ? current.filter((entry) => entry !== label)
        : [...current, label],
    );

  const submit = () => {
    const split = (value: string) =>
      value.split(",").map((entry) => entry.trim()).filter(Boolean);
    feedback.mutate({
      turn_id: turnId,
      kind: "correction",
      model,
      variables: picked,
      location: split(place),
      time: split(when),
      error_type: picked.join("|") !== variables.join("|")
        ? "intent_confusion"
        : split(place).join("|") !== locations.join("|")
          ? "location_resolution"
          : "time_resolution",
      note: note || undefined,
    });
    onDone();
  };

  return (
    <div className="mt-3 rounded-xl border bg-muted/40 p-3">
      <p className="text-xs font-medium">What should it have been?</p>
      <p className="mt-0.5 text-[10px] text-muted-foreground">
        Saved against this turn - sending it again updates the same record.
      </p>

      <div className="mt-2 flex flex-wrap gap-1.5">
        {options.map((label) => (
          <button
            key={label}
            onClick={() => toggle(label)}
            className={cn(
              "rounded-full border px-2 py-1 font-mono text-[10px] transition-colors",
              picked.includes(label)
                ? "border-primary/40 bg-primary text-primary-foreground"
                : "text-muted-foreground hover:bg-background",
            )}
          >
            {label}
          </button>
        ))}
      </div>

      <div className="mt-2.5 grid gap-2 sm:grid-cols-2">
        <label className="text-[11px] text-muted-foreground">
          Place
          <Input
            value={place}
            onChange={(event) => setPlace(event.target.value)}
            placeholder="Guntur"
            className="mt-1 h-8 text-xs"
          />
        </label>
        <label className="text-[11px] text-muted-foreground">
          When
          <Input
            value={when}
            onChange={(event) => setWhen(event.target.value)}
            placeholder="tomorrow"
            className="mt-1 h-8 text-xs"
          />
        </label>
      </div>

      <Input
        value={note}
        onChange={(event) => setNote(event.target.value)}
        placeholder="Anything else? (optional)"
        className="mt-2 h-8 text-xs"
      />

      <div className="mt-2.5 flex items-center gap-2">
        <Button size="sm" className="h-7 gap-1 text-xs" onClick={submit} disabled={!picked.length}>
          <Check className="h-3 w-3" />
          Save correction
        </Button>
        <Button size="sm" variant="ghost" className="h-7 gap-1 text-xs" onClick={onDone}>
          <X className="h-3 w-3" />
          Cancel
        </Button>
        <span className="text-[10px] text-muted-foreground">
          goes into the next training run
        </span>
      </div>
    </div>
  );
}
