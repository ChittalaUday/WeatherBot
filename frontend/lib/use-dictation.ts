"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { sttUrl } from "@/lib/utils";
const FLUSH_TIMEOUT_MS = 120_000;

const DELTA = "conversation.item.input_audio_transcription.delta";
const COMPLETED = "conversation.item.input_audio_transcription.completed";

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
  const stop = useRef<() => void>(() => { });
  const text = useRef(onText);
  useEffect(() => {
    text.current = onText;
  });

  const end = useCallback(() => {
    stop.current();
    stop.current = () => { };
    setRecording(false);
  }, []);

  const start = useCallback(async () => {
    setError(null);
    let stream: MediaStream | undefined;
    let ctx: AudioContext | undefined;
    let ws: WebSocket | undefined;
    let flush: ReturnType<typeof setTimeout> | undefined;
    let committed = false;

    let released = false;
    // Drop the mic. Idempotent, and it has to be: releasing the mic runs this, and so does
    // the teardown when the closing result lands a moment later. AudioContext.close()
    // throws on a context that is already closed or closing.
    const release = () => {
      if (released) return;
      released = true;
      setAnalyser(null);
      stream?.getTracks().forEach((track) => track.stop());
      if (ctx && ctx.state !== "closed") {
        try {
          void ctx.close().catch(() => {});
        } catch {
          // Ignore if AudioContext is already closing or closed
        }
      }
    };

    const teardown = () => {
      clearTimeout(flush);
      setPending(false);
      release();
      ws?.close();
    };

    stop.current = () => {
      // Release the mic straight away, but leave the socket open. The server replies at
      // its own pace - seconds per utterance, not milliseconds - and the words spoken just
      // before the release are still in flight; closing here would throw them away.
      release();
      if (ws?.readyState !== WebSocket.OPEN) return teardown();
      ws.send('{"type":"input_audio_buffer.commit"}');
      committed = true;
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
       ws.send('{"type":"session.update","session":{"language":"en-US","sample_rate":16000}}');

      const done: string[] = [];
      let partial = "";
      const emit = () => text.current([...done, partial].join(" ").replace(/\s+/g, " ").trim());

      ws.onmessage = (event) => {
        const message = JSON.parse(event.data as string);
        switch (message.type) {
          case DELTA:
            // Incremental, and split on token boundaries rather than words - "humidity"
            // arrives as " hum" + "id" + "ity" - so it is concatenated, not assigned.
            partial += message.delta ?? "";
            emit();
            break;
          case COMPLETED:
            // The final re-punctuates the whole utterance, so it replaces the deltas that
            // built it up rather than being appended to them.
            done.push(message.transcript ?? partial);
            partial = "";
            emit();
            // Nothing closes this socket for us: unlike the old stt/ service the server
            // stays open after committing, ready for another utterance.
            if (committed) teardown();
            break;
          case "error":
            setError(message.error?.message ?? "transcription failed");
            teardown();
            break;
        }
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
