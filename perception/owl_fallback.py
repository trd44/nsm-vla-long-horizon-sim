"""Optional runtime OWLv2 fallback for the NSM detector (driver flag --owl_fallback, off by default).

Paper section 3.4.2: OWLv2 detects the prompted objects "until the YOLOv8 model catches
up". At run time this wraps the YOLO model the executor calls. When a cube class the
plan needs has had no agentview YOLO box for ``n_missing`` consecutive agentview frames,
OWLv2 is queried once on CPU with the class prompts, and its boxes for the missing
classes are appended to that frame's YOLO result. Queries are rate limited
(``min_gap`` agentview frames apart, at most ``max_queries`` per episode), and counted
and timed per episode.

Nothing here reads simulator state. The executor's YOLO calls always come in pairs,
agentview then wrist (Executor_Diffusion.yolo_estimate), which is how the camera of
each call is known; only agentview frames trigger or receive OWL boxes.
"""
import time

import numpy as np

from perception.owl_label import CLASS_NAMES, IMG, LABEL_CFG

MAX_AGENT_PX = 40  # boxes larger than this are dropped by the executor anyway


class OwlFallbackModel:
    def __init__(self, yolo, n_missing=10, min_gap=10, max_queries=40, threads=1, device="cpu"):
        self.yolo = yolo
        self.names = yolo.names
        self.n_missing, self.min_gap, self.max_queries = n_missing, min_gap, max_queries
        self.threads, self.device = threads, device
        self.teacher = None
        self.load_s = 0.0
        self.name_to_cls = {v: k for k, v in CLASS_NAMES.items()}
        self.start_episode()

    def __getattr__(self, name):  # anything else the caller needs from the YOLO object
        return getattr(self.yolo, name)

    def start_episode(self, needed=None):
        """needed: YOLO class names the plan needs, or None to use every class YOLO has
        seen in agentview this episode (tasks where colour does not name the cube)."""
        self.needed = None if needed is None else {self.name_to_cls[n] for n in needed}
        self.seen = set()
        self.missing = {c: 0 for c in CLASS_NAMES}
        self.parity = 0
        self.frame = 0
        self.last_query = -10 ** 9
        self.stats = dict(queries=0, query_s=0.0, boxes_added=0, frames=0)

    def _teacher(self):
        if self.teacher is None:
            import torch

            from perception.owl_label import OwlTeacher
            t0 = time.time()
            prev = torch.get_num_threads()
            torch.set_num_threads(self.threads)
            self.teacher = OwlTeacher(self.device)
            torch.set_num_threads(prev)
            self.load_s = time.time() - t0
        return self.teacher

    def predict(self, image, **kwargs):
        res = self.yolo.predict(image, **kwargs)
        cam = "a" if self.parity == 0 else "w"
        self.parity ^= 1
        if cam != "a":
            return res
        r = res[0]
        self.frame += 1
        self.stats["frames"] += 1
        present = set()
        for b in r.boxes:
            x, y, w, h = (b.xywhn[0] * IMG).tolist()
            if max(w, h) <= MAX_AGENT_PX:
                present.add(int(b.cls))
        self.seen |= present
        needed = self.needed if self.needed is not None else self.seen
        for c in CLASS_NAMES:
            self.missing[c] = 0 if c in present else self.missing[c] + 1
        due = [c for c in needed if self.missing[c] >= self.n_missing]
        if (not due or self.frame - self.last_query < self.min_gap
                or self.stats["queries"] >= self.max_queries):
            return res
        import torch
        teacher = self._teacher()
        prev = torch.get_num_threads()
        torch.set_num_threads(self.threads)
        t0 = time.time()
        dets = teacher.detect(np.ascontiguousarray(image), "a", LABEL_CFG)
        torch.set_num_threads(prev)
        self.stats["query_s"] += time.time() - t0
        self.stats["queries"] += 1
        self.last_query = self.frame
        add = [d for d in dets if d[0] in due]
        if add:
            new = torch.tensor([[x0, y0, x1, y1, conf, c] for c, conf, x0, y0, x1, y1 in add], dtype=torch.float32)
            r.update(boxes=torch.cat([r.boxes.data.cpu().float(), new]) if len(r.boxes) else new)
            self.stats["boxes_added"] += len(add)
            for c, *_ in add:
                self.missing[c] = 0
        return res

    def episode_stats(self):
        return dict(owl_queries=self.stats["queries"], owl_query_s=round(self.stats["query_s"], 2),
                    owl_boxes_added=self.stats["boxes_added"], owl_agent_frames=self.stats["frames"],
                    owl_load_s=round(self.load_s, 2))
