import cv2
import jax
import bisect
import numpy as np
import os

class FrameExtractor:
    def __init__(self, directory_path, batch_size, key, target_size=(512,300)):
        self.directory_path = directory_path
        self.video_files = [f for f in os.listdir(directory_path) if f.endswith(('.mp4', '.avi'))] # Adjust as needed
        self.batch_size = batch_size
        self.key = key
        self.frame_counts = [int(cv2.VideoCapture(os.path.join(directory_path, f)).get(cv2.CAP_PROP_FRAME_COUNT)) for f in self.video_files]
        self.total_frames = sum(self.frame_counts)
        self.cumsum_frames = np.cumsum(self.frame_counts)
        self.cap = None
        self.target_size = target_size

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cap is not None:
            self.cap.release()

    def __iter__(self):
        return self

    def __next__(self):
        self.key, idx_key = jax.random.split(self.key) # split PRNG key into 2 new keys
        idx_array = jax.random.randint(idx_key, (self.batch_size,), 0, self.total_frames) # sample uniform random values in [0, self.total_frames)
        frames = []
        # global = across all videos, local = within a video
        for global_idx in idx_array:
            # find local index
            video_idx = bisect.bisect_right(self.cumsum_frames, global_idx) - 1
            local_idx = int(global_idx) - self.cumsum_frames[video_idx]
                
            self.cap = cv2.VideoCapture(os.path.join(self.directory_path, self.video_files[video_idx]))
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, local_idx)
            ret, frame = self.cap.read()
            self.cap.release()

            if ret:
                # resize video to specified target size
                # frame = cv2.resize(frame, self.target_size)
                frames.append(frame)

        array = jax.numpy.array(frames)
        return array.transpose(0,3,2,1)
