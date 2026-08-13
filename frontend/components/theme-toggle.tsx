"use client";

import { useTheme } from "next-themes";
import { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { MoonIcon } from "@/components/ui/moon-icon";
import { SunIcon } from "@/components/ui/sun-icon";

/** Light / dark, remembered by next-themes. Renders nothing until mounted, because the
 *  resolved theme is unknown on the server and a wrong first paint is worse than a beat of
 *  empty space. */
export function ThemeToggle() {
  const { resolvedTheme, setTheme } = useTheme();
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);

  const dark = resolvedTheme === "dark";
  return (
    <Button
      variant="ghost"
      size="sm"
      aria-label={dark ? "Switch to light mode" : "Switch to dark mode"}
      title={dark ? "Light mode" : "Dark mode"}
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
