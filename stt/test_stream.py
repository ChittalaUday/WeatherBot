"""Self-check for the session buffer. `python test_stream.py` (needs no model weights)."""
import threading

import numpy as np

from server import Stream

FRAME = np.full(2048, 1000, dtype=np.int16).tobytes()  # 2048 samples per browser message


def main():
    s = Stream()
    s.feed(FRAME)
    assert s.window(0, 2048) is not None, "a full window must come back"
    assert s.window(0, 2048).shape[0] == 2048, "window must be exactly the size asked for"

    # Windows overlap and are re-read, so feeding must not consume what was already handed out.
    s.feed(FRAME)
    assert s.window(1024, 3072) is not None, "an overlapping window must still be readable"
    assert s.audio.shape[0] == 4096, "the buffer keeps the whole session"

    # int16 -> float32 in [-1, 1]
    assert abs(s.audio[0] - 1000 / 32768) < 1e-6, "samples must be normalised"

    # A short session must not hang the model thread: close() has to release the waiter.
    s = Stream()
    s.feed(FRAME)
    result = []
    waiter = threading.Thread(target=lambda: result.append(s.window(0, 999_999)))
    waiter.start()
    waiter.join(0.2)
    assert waiter.is_alive(), "window() must block until the audio arrives"
    s.close()
    waiter.join(2)
    assert not waiter.is_alive(), "close() must wake the waiter"
    assert result == [None], "an unfillable window returns None so the generator can stop"

    # Blocking then satisfied by a later frame is the normal live path.
    s = Stream()
    result = []
    waiter = threading.Thread(target=lambda: result.append(s.window(0, 2048)))
    waiter.start()
    waiter.join(0.2)
    assert waiter.is_alive(), "must wait for audio that has not arrived yet"
    s.feed(FRAME)
    waiter.join(2)
    assert result[0] is not None and result[0].shape[0] == 2048, "late audio must satisfy it"

    print("stream ok")


if __name__ == "__main__":
    main()
