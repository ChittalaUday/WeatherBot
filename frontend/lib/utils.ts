import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

/**
 * Where the backend is. The single definition - eight files used to inline their own
 * `process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8787"`, so only one of them ever got
 * the LAN fallback below and the rest broke the moment the app was opened from another device.
 *
 * A configured URL wins, except a loopback one when the page itself is not on loopback: that
 * is a developer default left in a build someone is now reaching over the network, and the
 * page's own hostname is the answer.
 */
export function apiUrl(): string {
  const configured = process.env.NEXT_PUBLIC_API_URL;
  if (typeof window === "undefined") return configured ?? "http://127.0.0.1:8787";
  const loopback = /127\.0\.0\.1|localhost/;
  if (configured && !loopback.test(configured)) return configured;
  if (loopback.test(window.location.hostname)) return configured ?? "http://127.0.0.1:8787";
  return `${window.location.protocol}//${window.location.hostname}:8787`;
}

/**
 * The streaming speech-to-text socket (the Nemotron ASR container in stt/). Same fallback rule
 * as apiUrl(): a loopback default is ignored when the page is being reached over the network.
 */
export function sttUrl(): string {
  const configured = process.env.NEXT_PUBLIC_STT_URL;
  const loopback = /127\.0\.0\.1|localhost/;
  if (configured && !(loopback.test(configured) && !loopback.test(window.location.hostname))) {
    return configured;
  }
  return `ws://${window.location.hostname}:2700`;
}

/**
 * The speech service's health endpoint, served beside the socket on the same port.
 *
 * Deliberately not the backend's /api/health: the browser talks to the speech service
 * directly, so only the speech service's own origin can answer whether dictation will
 * work from here. ws:// -> http://, wss:// -> https://.
 */
export function sttHealthUrl(): string {
  return `${sttUrl().replace(/^ws/, "http").replace(/\/$/, "")}/health`;
}
