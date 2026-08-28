"use client";

import { useMutation } from "@tanstack/react-query";
import { apiUrl } from "@/lib/utils";


export type FeedbackBody = {
  turn_id: number;
  kind: "up" | "down" | "correction" | "choice";
  intent?: string;
  action?: string;
  variables?: string[];
  location?: string[];
  time?: string[];
  model?: string;
  error_type?: string;
  note?: string;
};

/** Sends a label to backend/store.py, where it waits to be folded into the next training run. */
export function useFeedback() {
  return useMutation({
    mutationFn: async (body: FeedbackBody) => {
      if (!Number.isInteger(body.turn_id)) throw new Error("this answer has no turn to rate");
      const response = await fetch(`${apiUrl()}/api/feedback`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!response.ok) throw new Error(`feedback rejected (${response.status})`);
      return response.json();
    },
  });
}
