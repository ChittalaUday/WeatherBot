"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { sttUrl } from "@/lib/utils";

/**
 * Mic -> the Nemotron ASR container (stt/) -> text, as you speak.
 *
 * The server runs a cache-aware streaming model, so {"partial"} lands roughly per word
 * while you are still talking, and {"text"} closes the utterance.
 *
 * The server wants 16 kHz mono PCM s16le, so the AudioContext is opened at 16 kHz and the
 * browser resamples the mic for us - no hand-written resampler. The worklet is a blob so
 * there is no extra file to serve from public/, and it converts to Int16 and batches 2048
 * samples (128 ms) per message; posting every 128-sample render quantum floods the socket.
 *
 * onText is called with the whole transcript so far (finished utterances plus the partial
 * being spoken), so a caller can just drop it into its input.
 *
 * `pending` covers the gap between releasing the mic and the server's closing result -
 * short, but the last words land in it, so the UI must not look finished yet.
 *
 * `analyser` is a tap on the mic for drawing a live trace; it is null unless recording.
 */

// Only a safety net for a server that never answers: the normal path ends when the server
// sends its last result and closes, which is prompt. Generous on purpose - a slow host
// finishing a long utterance must not have its transcript cut off.
const FLUSH_TIMEOUT_MS = 120_000;

const WORKLET = `
class PCM extends AudioWorkletProcessor {
  buf = new Int16Array(2048);
  n = 0;
  process(inputs) {
    const ch = inputs[0][0];
    if (!ch) return true;
    for (let i = 0; i < ch.length; i++) {
      this.buf[this.n++] = Math.max(-1, Math.min(1, ch[i])) * 0x7fff;
      if (this.n === this.buf.length) { this.port.postMessage(this.buf.slice()); this.n = 0; }
    }
    return true;
  }
}
registerProcessor("pcm", PCM);
`;

export function useDictation(onText: (text: string) => void) {
  const [recording, setRecording] = useState(false);
  const [pending, setPending] = useState(false);
  const [analyser, setAnalyser] = useState<AnalyserNode | null>(null);
  const [error, setError] = useState<string | null>(null);
  const stop = useRef<() => void>(() => {});
  const text = useRef(onText);
  useEffect(() => {
    text.current = onText;
  });

  const end = useCallback(() => {
    stop.current();
    stop.current = () => {};
    setRecording(false);
  }, []);

  const start = useCallback(async () => {
    setError(null);
    let stream: MediaStream | undefined;
    let ctx: AudioContext | undefined;
    let ws: WebSocket | undefined;
    let flush: ReturnType<typeof setTimeout> | undefined;

    const teardown = () => {
      clearTimeout(flush);
      setPending(false);
      setAnalyser(null);
      stream?.getTracks().forEach((track) => track.stop());
      void ctx?.close();
      ws?.close();
    };

    stop.current = () => {
      // Release the mic straight away, but leave the socket open. The server replies at
      // its own pace - seconds per utterance, not milliseconds - and closes once it has
      // flushed the last one; closing here would throw that transcript away.
      setAnalyser(null);
      stream?.getTracks().forEach((track) => track.stop());
      void ctx?.close();
      if (ws?.readyState !== WebSocket.OPEN) return teardown();
      ws.send('{"eof":1}');
      setPending(true);
      flush = setTimeout(teardown, FLUSH_TIMEOUT_MS);
    };

    try {
      stream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
      });
      ctx = new AudioContext({ sampleRate: 16000 });
      await ctx.audioWorklet.addModule(URL.createObjectURL(new Blob([WORKLET], { type: "text/javascript" })));
      ws = new WebSocket(sttUrl());
      ws.binaryType = "arraybuffer";
      await new Promise<void>((resolve, reject) => {
        ws!.onopen = () => resolve();
        ws!.onerror = () => reject(new Error("speech service unreachable"));
      });

      const done: string[] = [];
      ws.onmessage = (event) => {
        const message = JSON.parse(event.data as string);
        if (message.text) done.push(message.text);
        text.current([...done, message.partial ?? ""].join(" ").trim());
      };
      ws.onclose = () => {
        clearTimeout(flush);
        setPending(false);
        end();
      };

      const node = new AudioWorkletNode(ctx, "pcm");
      node.port.onmessage = (event) => {
        if (ws?.readyState === WebSocket.OPEN) ws.send(event.data as Int16Array);
      };
      const mute = ctx.createGain();
      mute.gain.value = 0;
      // Chrome only pulls audio through a worklet that reaches the destination; the gain
      // node keeps the mic out of the speakers.
      const source = ctx.createMediaStreamSource(stream);
      source.connect(node).connect(mute).connect(ctx.destination);
      // A dead-end tap for the waveform: the analyser reads the mic without being wired
      // onward, so drawing cannot affect what gets sent.
      const tap = ctx.createAnalyser();
      tap.fftSize = 1024;
      source.connect(tap);
      setAnalyser(tap);
      setRecording(true);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "microphone unavailable");
      stop.current = teardown;
      end();
    }
  }, [end]);

  useEffect(() => end, [end]);

  return { recording, pending, error, analyser, toggle: () => (recording ? end() : void start()) };
}
