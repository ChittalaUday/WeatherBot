"use client";

import { useEffect, useRef } from "react";

/**
 * Live mic trace drawn from AnalyserNode as a marquee-scrolling audio tape waveform.
 *
 * Features:
 * - Continuous marquee scroll with 100% continuous position math.
 * - Sub-pixel progress interpolation for 60fps/120fps glass-smooth motion.
 * - High sensitivity gain & noise floor gating.
 * - Edge fade-out & resting baseline dots matching modern voice overlays.
 */

const TOTAL_BARS = 64; // Number of historical bars stored in the buffer
const MIN_BAR = 2.5; // Baseline px height for resting silent dots
const NOISE_FLOOR = 0.015; // Low threshold so voice registers immediately
const SAMPLE_INTERVAL_MS = 100; // Calmer, smooth scrolling speed (~10 bars/sec)

export function Waveform({
  analyser,
  className,
}: {
  analyser: AnalyserNode;
  className?: string;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Buffer storing past bar height amplitudes (0.0 to 1.0)
    const history = new Float32Array(TOTAL_BARS);

    let animationFrameId: number;
    let resizeObserver: ResizeObserver | null = null;

    let barColor = getComputedStyle(canvas).color || "#ef4444";

    const updateDimensions = () => {
      const dpr = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      if (rect.width === 0 || rect.height === 0) return;
      canvas.width = Math.floor(rect.width * dpr);
      canvas.height = Math.floor(rect.height * dpr);
      barColor = getComputedStyle(canvas).color || "#ef4444";
    };

    updateDimensions();

    if (typeof ResizeObserver !== "undefined") {
      resizeObserver = new ResizeObserver(updateDimensions);
      resizeObserver.observe(canvas);
    }

    const samples = new Uint8Array(analyser.fftSize);
    let smoothedVolume = 0;
    let lastSampleTime = performance.now();

    const draw = (now: number) => {
      animationFrameId = requestAnimationFrame(draw);

      const dpr = window.devicePixelRatio || 1;
      const width = canvas.width;
      const height = canvas.height;

      if (width === 0 || height === 0) return;

      // 1. Read audio time domain data & calculate current instantaneous peak
      analyser.getByteTimeDomainData(samples);
      let rawPeak = 0;
      for (let i = 0; i < samples.length; i++) {
        const val = Math.abs(samples[i] - 128) / 128;
        if (val > rawPeak) rawPeak = val;
      }

      // Noise floor gate
      let gatedVal = 0;
      if (rawPeak > NOISE_FLOOR) {
        gatedVal = (rawPeak - NOISE_FLOOR) / (1 - NOISE_FLOOR);
      }

      // Boosted gain curve for high sensitivity (sub-1 exponent + 2.2x gain)
      const boosted = Math.pow(gatedVal, 0.85) * 2.2;
      const targetVolume = Math.min(1.0, boosted);

      // Attack / release volume smoothing
      const speed = targetVolume > smoothedVolume ? 0.25 : 0.12;
      smoothedVolume += (targetVolume - smoothedVolume) * speed;

      // 2. Advance scrolling history timer
      const elapsed = now - lastSampleTime;
      if (elapsed >= SAMPLE_INTERVAL_MS) {
        const steps = Math.floor(elapsed / SAMPLE_INTERVAL_MS);
        lastSampleTime += steps * SAMPLE_INTERVAL_MS;

        for (let s = 0; s < steps; s++) {
          // Shift history buffer to the left
          for (let i = 0; i < TOTAL_BARS - 1; i++) {
            history[i] = history[i + 1];
          }
          // Blend with previous bar for continuous wave profile
          const prevVal = history[TOTAL_BARS - 2] || 0;
          history[TOTAL_BARS - 1] = prevVal * 0.4 + smoothedVolume * 0.6;
        }
      }

      // Continuous progress fraction between [0.0, 1.0]
      const progress = Math.min(1.0, Math.max(0.0, (now - lastSampleTime) / SAMPLE_INTERVAL_MS));

      // 3. Render canvas
      ctx.clearRect(0, 0, width, height);

      const barGap = 2 * dpr;
      const barWidth = 2 * dpr;
      const step = barWidth + barGap;

      const minH = MIN_BAR * dpr;
      const maxH = height * 0.92;
      const fadeWidth = 24 * dpr;

      for (let i = 0; i < TOTAL_BARS; i++) {
        const amplitude = history[i];
        const barH = minH + amplitude * (maxH - minH);

        // Continuous smooth position math:
        // x moves continuously to the left as progress increases from 0 to 1
        const x = width - (TOTAL_BARS - 1 - i + progress) * step;

        // Skip off-screen bars
        if (x + barWidth < 0 || x > width) continue;

        const y = (height - barH) / 2;
        const radius = Math.min(barWidth / 2, barH / 2);

        // Edge opacity fade out (left and right edges)
        let alpha = 1.0;
        if (x < fadeWidth) {
          alpha = Math.max(0, x / fadeWidth);
        } else if (x > width - fadeWidth) {
          alpha = Math.max(0, (width - x) / fadeWidth);
        }

        ctx.fillStyle = barColor;
        ctx.globalAlpha = alpha;

        ctx.beginPath();
        ctx.roundRect(x, y, barWidth, barH, radius);
        ctx.fill();
      }

      // Reset alpha
      ctx.globalAlpha = 1.0;
    };

    animationFrameId = requestAnimationFrame(draw);

    return () => {
      cancelAnimationFrame(animationFrameId);
      if (resizeObserver) resizeObserver.disconnect();
    };
  }, [analyser]);

  return <canvas ref={canvasRef} aria-hidden className={className} />;
}




