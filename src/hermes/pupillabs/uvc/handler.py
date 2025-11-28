############
#
# Copyright (c) 2024-2025 Maxim Yudayev and KU Leuven eMedia Lab
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Created 2024-2025 for the KU Leuven AidWear, AidFOG, and RevalExo projects
# by Maxim Yudayev [https://yudayev.com].
#
# ############

import time
from typing import Callable
import uvc
from multiprocessing import Queue, Event

from hermes.utils.time_utils import get_time, init_time
from hermes.utils.types import VideoFormatEnum


class PupilUvcHandler:
    def __call__(
        self,
        ref_time_s: float,
        camera_name: str,
        camera_spec: dict,
        queue: Queue,
        video_image_format: VideoFormatEnum,
        stop_event: Event,
        keep_event: Event,
        ready_event: Event,
    ):
        init_time(ref_time_s)
        self.camera_name = camera_name
        self.camera_spec = camera_spec
        self.queue = queue
        self.cap: uvc.Capture

        self._restart_cap_device()

        if video_image_format == VideoFormatEnum.MJPEG:
            get_buffer_fn = lambda frame: bytes(frame.jpeg_buffer)
        elif video_image_format == VideoFormatEnum.BGR:
            get_buffer_fn = lambda frame: frame.bgr
        elif video_image_format == VideoFormatEnum.YUV:
            get_buffer_fn = lambda frame: frame.yuv
        else:
            get_buffer_fn = lambda _: None

        # synchronize worker process to the upstream `keep_data` signal.
        ready_event.set()
        keep_event.wait()

        while not stop_event.is_set():
            self._get_frame(get_buffer_fn)
        self.cap.close()


    def _restart_cap_device(self):
        try:
            devices = dict(map(lambda dev: (dev["name"], dev["uid"]), uvc.device_list()))

            self.cap = uvc.Capture(devices[self.camera_spec["name"]])
            self.cap.bandwidth_factor = self.camera_spec["bandwidth_factor"]

            for mode in self.cap.available_modes:
                if (
                    mode.width == self.camera_spec["resolution"][1]
                    and mode.height == self.camera_spec["resolution"][0]
                    and mode.fps == self.camera_spec["fps"]
                ):
                    self.cap.frame_mode = mode
                    break
                # configure the controls on each `Capture` object (exposure, brightness, sharpness, etc).
                controls_by_name = {c.display_name: c for c in self.cap.controls}

            print(f"Settings controls for {self.camera_spec['name']}", flush=True)
            for ctrl_name, value in self.camera_spec.get("uvc_controls", {}).items():
                ctrl = controls_by_name.get(ctrl_name)
                try:
                    ctrl.value = value
                except Exception as e:
                    print(f"Could not set control for {self.camera_spec['name']} '{ctrl_name}' to {value}: {e}", file=True)
        except Exception as e:
            time.sleep(1)


    def _get_frame(self, get_buffer_fn: Callable) -> None:
        try:
            frame = self.cap.get_frame(timeout=1)
            toa_s = get_time()
            out = {
                "timestamp": frame.timestamp,
                "index": frame.index,
                "data": get_buffer_fn(frame),
            }
            self.queue.put((self.camera_name, out, toa_s))
        except uvc.InitError as err:
            print(f"[PupilUvcProducer] Failed to init {self.camera_name}: {err}", flush=True)
        except uvc.StreamError as err:
            print(f"[PupilUvcProducer] Stream error for {self.camera_name}: {err}", flush=True)
        except (TimeoutError, NameError, AttributeError) as err:
            print(f"[PupilUvcProducer] Restarting lost connection to '{self.camera_name}' camera: {err}", flush=True)
            self._restart_cap_device()
