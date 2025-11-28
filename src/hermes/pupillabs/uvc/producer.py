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

from multiprocessing import Process, Queue, Event
from queue import Empty

from hermes.utils.mp_utils import launch_callable
from hermes.utils.zmq_utils import PORT_BACKEND, PORT_SYNC_HOST, PORT_KILL
from hermes.utils.time_utils import get_time
from hermes.utils.types import VideoFormatEnum, LoggingSpec

from hermes.base.nodes.producer import Producer

from .stream import PupilUvcStream
from .handler import PupilUvcHandler


class PupilUvcProducer(Producer):
    @classmethod
    def _log_source_tag(cls) -> str:
        return "glasses"

    def __init__(
        self,
        host_ip: str,
        camera_mapping: dict,
        logging_spec: LoggingSpec,
        video_image_format: VideoFormatEnum = VideoFormatEnum.MJPEG,
        port_pub: str = PORT_BACKEND,
        port_sync: str = PORT_SYNC_HOST,
        port_killsig: str = PORT_KILL,
        transmit_delay_sample_period_s: float = float("nan"),
        timesteps_before_solidified: int = 0,
        **_,
    ):
        self._camera_mapping = camera_mapping
        self._video_image_format = video_image_format
        self._is_continue_grabbing = True
        self._start_index: dict[str, int | None] = dict(
            map(lambda cam: (cam, None), self._camera_mapping.keys())
        )
        self._parse_frame_fn = self._parse_first_frame
        self._cap_queue: Queue = Queue()
        self._cap_procs: list[Process] = []
        self._cap_handlers: list[PupilUvcHandler] = []
        self._stop_event = Event()
        self._keep_event = Event()

        stream_out_spec = {
            "camera_mapping": self._camera_mapping,
            "pixel_format": video_image_format,
            "timesteps_before_solidified": timesteps_before_solidified,
        }

        super().__init__(
            host_ip=host_ip,
            stream_out_spec=stream_out_spec,
            logging_spec=logging_spec,
            port_pub=port_pub,
            port_sync=port_sync,
            port_killsig=port_killsig,
            transmit_delay_sample_period_s=transmit_delay_sample_period_s,
        )

    @classmethod
    def create_stream(cls, stream_info: dict) -> PupilUvcStream:
        return PupilUvcStream(**stream_info)

    def _ping_device(self) -> None:
        return None

    def _connect(self) -> bool:
        # launch each capture subprocess
        ready_events: list[Event] = []
        for cam in self._camera_mapping.keys():
            handler = PupilUvcHandler()
            ready_event = Event()
            proc = Process(
                target=launch_callable,
                args=(
                    handler,
                    self._ref_time_s,
                    cam,
                    self._camera_mapping[cam],
                    self._cap_queue,
                    self._video_image_format,
                    self._stop_event,
                    self._keep_event,
                    ready_event,
                )
            )
            self._cap_procs.append(proc)
            self._cap_handlers.append(handler)
            ready_events.append(ready_event)
            proc.start()

        # read their pipe with confirmation that each connected to the camera
        for event in ready_events: event.wait()
        return True

    def _keep_samples(self) -> None:
        self._keep_event.set()

    def _process_data(self) -> None:
        try:
            msg = self._cap_queue.get(timeout=10)
            process_time_s = get_time()
            output = self._parse_frame_fn(msg)
            if output is not None:
                tag: str = "%s.data" % self._log_source_tag()
                self._publish(tag, process_time_s=process_time_s, data=output)
        except Empty:
            if not self._is_continue_capture:
                self._send_end_packet()

    def _parse_first_frame(self, msg: tuple) -> dict:
        camera_name, frame, toa_s = msg

        if self._start_index[camera_name] is None:
            self._start_index[camera_name] = frame["index"]
            frame_index = 0
        else:
            frame_index = frame["index"] - self._start_index[camera_name]

        if all(v is not None for v in self._start_index.values()):
            self._parse_frame_fn = self._parse_frame

        output: dict[str, dict] = {}
        output[camera_name] = {
            "frame_timestamp": frame["timestamp"],
            "frame_index": frame_index,
            "frame_sequence_id": frame["index"],
            "frame": (frame["data"], False, frame_index),
            "toa_s": toa_s,
        }
        return output

    def _parse_frame(self, msg: tuple) -> dict:
        camera_name, frame, toa_s = msg

        frame_index = frame["index"] - self._start_index[camera_name]

        output: dict[str, dict] = {}
        output[camera_name] = {
            "frame_timestamp": frame["timestamp"],
            "frame_index": frame_index,
            "frame_sequence_id": frame["index"],
            "frame": (frame["data"], False, frame_index),
            "toa_s": toa_s,
        }
        return output

    def _stop_new_data(self) -> None:
        self._stop_event.set()

    def _cleanup(self) -> None:
        for proc in self._cap_procs:
            proc.join()
        super()._cleanup()
