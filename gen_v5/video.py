"""Per-step video: frames sampled from the 20 Hz control loop, written as H.264 mp4 (as RoboJEPA's loaders expect)."""
import av
import numpy as np


def write_mp4(path, frames, fps, crf=23):
    frames = np.asarray(frames, np.uint8)
    with av.open(str(path), "w") as c:
        s = c.add_stream("libx264", rate=int(fps))
        s.width, s.height = frames.shape[2], frames.shape[1]
        s.pix_fmt = "yuv420p"
        s.options = {"crf": str(crf), "preset": "medium"}
        for f in frames:
            for p in s.encode(av.VideoFrame.from_ndarray(f, format="rgb24")):
                c.mux(p)
        for p in s.encode():
            c.mux(p)


def read_mp4(path):
    with av.open(str(path)) as c:
        return np.stack([f.to_ndarray(format="rgb24") for f in c.decode(video=0)])
