"use client";

import { useEffect, useRef } from "react";

/**
 * Live mic trace, drawn straight from the AnalyserNode on every animation frame.
 *
 * The samples never go through React state - at 60fps that would re-render the whole
 * composer - so this reads the analyser inside the rAF loop and paints a canvas.
 * Bars are symmetric about the middle, which reads as a voice level rather than a
 * scope trace, and it takes its colour from `currentColor` so the caller themes it.
 */

const BARS = 24;
const MIN_BAR = 2; // css px, so silence is a flat line of dots rather than nothing

export function Waveform({
  analyser,
  className,
}: {
  analyser: AnalyserNode;
  className?: string;
}) {
  const canvas = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const element = canvas.current;
    const context = element?.getContext("2d");
    if (!element || !context) return;

    const dpr = window.devicePixelRatio || 1;
    element.width = element.clientWidth * dpr;
    element.height = element.clientHeight * dpr;
    // Read once: recording lasts seconds, so a theme flip mid-trace is not worth a
    // getComputedStyle on every frame.
    context.fillStyle = getComputedStyle(element).color;

    const samples = new Uint8Array(analyser.fftSize);
    const slice = Math.floor(samples.length / BARS);
    const width = element.width / (BARS * 2 - 1); // bar + one gap of equal width
    let frame = 0;

    const draw = () => {
      frame = requestAnimationFrame(draw);
      analyser.getByteTimeDomainData(samples);
      context.clearRect(0, 0, element.width, element.height);
      for (let bar = 0; bar < BARS; bar++) {
        let peak = 0;
        for (let i = bar * slice; i < (bar + 1) * slice; i++) {
          peak = Math.max(peak, Math.abs(samples[i] - 128) / 128);
        }
        const height = Math.max(MIN_BAR * dpr, Math.min(1, peak * 2) * element.height);
        const x = bar * width * 2;
        const y = (element.height - height) / 2;
        context.beginPath();
        context.roundRect(x, y, width, height, width / 2);
        context.fill();
      }
    };
    draw();

    return () => cancelAnimationFrame(frame);
  }, [analyser]);

  return <canvas ref={canvas} aria-hidden className={className} />;
}
