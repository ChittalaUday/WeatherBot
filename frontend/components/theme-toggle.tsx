"use client";

import { useTheme } from "next-themes";
import { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { MoonIcon } from "@/components/ui/moon-icon";
import { SunIcon } from "@/components/ui/sun-icon";

/**
 * Light / dark, remembered by next-themes.
 *
 * Everything theme-derived - icon, label, title - waits for mount. The server cannot know
 * the resolved theme, so rendering "Switch to light mode" there and "Switch to dark mode"
 * on the client is a hydration mismatch, not just a wrong-looking icon.
 */
export function ThemeToggle() {
  const { resolvedTheme, setTheme } = useTheme();
  const [mounted, setMounted] = useState(false);
  // next-themes' documented hydration guard, and the one case the lint rule cannot allow for:
  // the flag has to flip *after* the first client render or the server's markup and the
  // client's disagree. There is nothing to synchronise it with but the render itself.
  // eslint-disable-next-line react-hooks/set-state-in-effect
  useEffect(() => setMounted(true), []);

  const dark = mounted && resolvedTheme === "dark";
  const label = mounted ? (dark ? "Switch to light mode" : "Switch to dark mode") : "Toggle theme";

  return (
    <Button
      variant="ghost"
      size="sm"
      aria-label={label}
      title={label}
      onClick={() => setTheme(dark ? "light" : "dark")}
      className="h-8 w-8 px-0 text-muted-foreground"
    >
      {mounted ? (
        dark ? <SunIcon size={15} isAnimated /> : <MoonIcon size={15} isAnimated />
      ) : (
        <span className="h-4 w-4" />
      )}
    </Button>
  );
}
