############
#
# Copyright (c) 2024-2026 Maxim Yudayev and KU Leuven eMedia Lab
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

from collections import OrderedDict
from typing import Optional

from hermes.base.data_container import DataContainer
from hermes.utils.types import VideoFormatEnum


class PupilCoreDataContainer(DataContainer):
    """A structure to store Pupil Core `Node`s data."""

    def __init__(
        self,
        is_binocular: bool,
        is_stream_video_world: bool,
        is_stream_video_eye: bool,
        is_stream_fixation: bool,
        is_stream_blinks: bool,
        gaze_estimate_stale_s: float,
        shape_video_world: tuple,
        shape_video_eye0: tuple,
        shape_video_eye1: tuple,
        fps_video_world: float,
        fps_video_eye0: float,
        fps_video_eye1: float,
        pixel_format: VideoFormatEnum,
        buf_len: Optional[int] = 3000,
        mem_size: Optional[int] = 100000000,
        timesteps_before_solidified: Optional[int] = 0,
        update_interval_ms: Optional[int] = 100,
        **_
    ) -> None:
        super().__init__()

        self._is_binocular = is_binocular
        self._is_stream_video_world = is_stream_video_world
        self._is_stream_video_eye = is_stream_video_eye
        self._is_stream_fixation = is_stream_fixation
        self._is_stream_blinks = is_stream_blinks
        self._gaze_estimate_stale_s = gaze_estimate_stale_s
        self._update_interval_ms = update_interval_ms
        self._pixel_format = pixel_format
        self._timesteps_before_solidified = timesteps_before_solidified

        # Define data notes that will be associated with streams created below.
        self._define_data_notes()

        # Create a stream for the Pupil Core time, to help evaluate drift and offsets.
        # Note that core time is included with each other stream as well,
        #  but include a dedicated one too just in case there are delays in sending
        #  the other data payloads.
        self.add_channel(
            bundle_name="eye_time",
            channel_name="device_time_s",
            data_type="float64",
            sample_size=[1],
            buf_len=buf_len,
            sampling_rate_hz=fps_video_world,
            data_notes=self._data_notes["eye_time"]["device_time_s"],
        )
        self.add_channel(
            bundle_name="eye_time",
            channel_name="toa_s",
            data_type="float64",
            sample_size=[1],
            buf_len=buf_len,
            sampling_rate_hz=fps_video_world,
        )

        # Create streams for gaze data.
        self.add_channel(
            bundle_name="eye_gaze",
            channel_name="confidence",
            data_type="float64",
            sample_size=[1],
            sampling_rate_hz=fps_video_world,
            data_notes=self._data_notes["eye_gaze"]["confidence"],
        )
        self.add_channel(
            bundle_name="eye_gaze",
            channel_name="eye_center_3d",
            data_type="float64",
            sample_size=[2, 3],
            buf_len=buf_len,
            sampling_rate_hz=fps_video_world,
            data_notes=self._data_notes["eye_gaze"]["eye_center_3d"],
        )
        self.add_channel(
            bundle_name="eye_gaze",
            channel_name="normal_3d",
            data_type="float64",
            sample_size=[2, 3],
            buf_len=buf_len,
            sampling_rate_hz=fps_video_world,
            data_notes=self._data_notes["eye_gaze"]["normal_3d"],
        )
        self.add_channel(
            bundle_name="eye_gaze",
            channel_name="point_3d",
            data_type="float64",
            sample_size=[3],
            buf_len=buf_len,
            sampling_rate_hz=fps_video_world,
            data_notes=self._data_notes["eye_gaze"]["point_3d"],
        )
        self.add_channel(
            bundle_name="eye_gaze",
            channel_name="position",
            data_type="float64",
            sample_size=[2],
            buf_len=buf_len,
            sampling_rate_hz=fps_video_world,
            data_notes=self._data_notes["eye_gaze"]["position"],
        )
        self.add_channel(
            bundle_name="eye_gaze",
            channel_name="timestamp",
            data_type="float64",
            sample_size=[1],
            buf_len=buf_len,
            sampling_rate_hz=fps_video_world,
            is_measure_rate_hz=True,
            data_notes=self._data_notes["eye_gaze"]["timestamp"],
        )

        # Create streams for pupil data.
        self.add_channel(
            bundle_name="eye_pupil",
            channel_name="confidence",
            data_type="float64",
            sample_size=[2],
            buf_len=buf_len,
            sampling_rate_hz=fps_video_world,
            data_notes=self._data_notes["eye_pupil"]["confidence"],
        )
        self.add_channel(
            bundle_name="eye_pupil",
            channel_name="circle3d_center",
            data_type="float64",
            sample_size=[2, 3],
            buf_len=buf_len,
            sampling_rate_hz=fps_video_world,
            data_notes=self._data_notes["eye_pupil"]["circle3d_center"],
        )
        self.add_channel(
            bundle_name="eye_pupil",
            channel_name="circle3d_normal",
            data_type="float64",
            sample_size=[2, 3],
            buf_len=buf_len,
            sampling_rate_hz=fps_video_world,
            data_notes=self._data_notes["eye_pupil"]["circle3d_normal"],
        )
        self.add_channel(
            bundle_name="eye_pupil",
            channel_name="circle3d_radius",
            data_type="float64",
            sample_size=[2],
            buf_len=buf_len,
            sampling_rate_hz=fps_video_world,
            data_notes=self._data_notes["eye_pupil"]["circle3d_radius"],
        )
        self.add_channel(
            bundle_name="eye_pupil",
            channel_name="diameter",
            data_type="float64",
            sample_size=[2],
            buf_len=buf_len,
            sampling_rate_hz=fps_video_world,
            data_notes=self._data_notes["eye_pupil"]["diameter"],
        )
        self.add_channel(
            bundle_name="eye_pupil",
            channel_name="diameter3d",
            data_type="float64",
            sample_size=[2],
            buf_len=buf_len,
            sampling_rate_hz=fps_video_world,
            data_notes=self._data_notes["eye_pupil"]["diameter3d"],
        )
        self.add_channel(
            bundle_name="eye_pupil",
            channel_name="polar_phi",
            data_type="float64",
            sample_size=[2],
            buf_len=buf_len,
            sampling_rate_hz=fps_video_world,
            data_notes=self._data_notes["eye_pupil"]["polar_phi"],
        )
        self.add_channel(
            bundle_name="eye_pupil",
            channel_name="polar_theta",
            data_type="float64",
            sample_size=[2],
            buf_len=buf_len,
            sampling_rate_hz=fps_video_world,
            data_notes=self._data_notes["eye_pupil"]["polar_theta"],
        )
        self.add_channel(
            bundle_name="eye_pupil",
            channel_name="position",
            data_type="float64",
            sample_size=[2, 2],
            buf_len=buf_len,
            sampling_rate_hz=fps_video_world,
            data_notes=self._data_notes["eye_pupil"]["position"],
        )
        self.add_channel(
            bundle_name="eye_pupil",
            channel_name="projected_sphere_angle",
            data_type="float64",
            sample_size=[2],
            buf_len=buf_len,
            sampling_rate_hz=fps_video_world,
            data_notes=self._data_notes["eye_pupil"]["projected_sphere_angle"],
        )
        self.add_channel(
            bundle_name="eye_pupil",
            channel_name="projected_sphere_axes",
            data_type="float64",
            sample_size=[2, 2],
            buf_len=buf_len,
            sampling_rate_hz=fps_video_world,
            data_notes=self._data_notes["eye_pupil"]["projected_sphere_axes"],
        )
        self.add_channel(
            bundle_name="eye_pupil",
            channel_name="projected_sphere_center",
            data_type="float64",
            sample_size=[2, 2],
            buf_len=buf_len,
            sampling_rate_hz=fps_video_world,
            data_notes=self._data_notes["eye_pupil"]["projected_sphere_center"],
        )
        self.add_channel(
            bundle_name="eye_pupil",
            channel_name="sphere_center",
            data_type="float64",
            sample_size=[2, 3],
            buf_len=buf_len,
            sampling_rate_hz=fps_video_world,
            data_notes=self._data_notes["eye_pupil"]["sphere_center"],
        )
        self.add_channel(
            bundle_name="eye_pupil",
            channel_name="sphere_radius",
            data_type="float64",
            sample_size=[2],
            buf_len=buf_len,
            sampling_rate_hz=fps_video_world,
            data_notes=self._data_notes["eye_pupil"]["sphere_radius"],
        )
        self.add_channel(
            bundle_name="eye_pupil",
            channel_name="timestamp",
            data_type="float64",
            sample_size=[2],
            buf_len=buf_len,
            sampling_rate_hz=fps_video_world,
            is_measure_rate_hz=True,
            data_notes=self._data_notes["eye_pupil"]["timestamp"],
        )

        # Create streams for fixation data.
        if is_stream_fixation:
            self.add_channel(
                bundle_name="eye_fixations",
                channel_name="id",
                data_type="uint64",
                sample_size=[2],
                buf_len=buf_len,
                data_notes=self._data_notes["eye_fixations"]["id"],
            )
            self.add_channel(
                bundle_name="eye_fixations",
                channel_name="timestamp",
                data_type="float64",
                sample_size=[1],
                buf_len=buf_len,
                data_notes=self._data_notes["eye_fixations"]["timestamp"],
            )
            self.add_channel(
                bundle_name="eye_fixations",
                channel_name="norm_pos",
                data_type="float64",
                sample_size=[2],
                buf_len=buf_len,
                data_notes=self._data_notes["eye_fixations"]["norm_pos"],
            )
            self.add_channel(
                bundle_name="eye_fixations",
                channel_name="dispersion",
                data_type="float64",
                sample_size=[1],
                buf_len=buf_len,
                data_notes=self._data_notes["eye_fixations"]["dispersion"],
            )
            self.add_channel(
                bundle_name="eye_fixations",
                channel_name="duration",
                data_type="float64",
                sample_size=[1],
                buf_len=buf_len,
                data_notes=self._data_notes["eye_fixations"]["duration"],
            )
            self.add_channel(
                bundle_name="eye_fixations",
                channel_name="confidence",
                data_type="float64",
                sample_size=[1],
                buf_len=buf_len,
                data_notes=self._data_notes["eye_fixations"]["confidence"],
            )
            self.add_channel(
                bundle_name="eye_fixations",
                channel_name="gaze_point_3d",
                data_type="float64",
                sample_size=[3],
                buf_len=buf_len,
                data_notes=self._data_notes["eye_fixations"]["gaze_point_3d"],
            )

        # Create streams for blinks data.
        if is_stream_blinks:
            self.add_channel(
                bundle_name="eye_blinks",
                channel_name="timestamp",
                data_type="float64",
                sample_size=[1],
                buf_len=buf_len,
                data_notes=self._data_notes["eye_blinks"]["timestamp"],
            )
            self.add_channel(
                bundle_name="eye_blinks",
                channel_name="confidence",
                data_type="float64",
                sample_size=[1],
                buf_len=buf_len,
                data_notes=self._data_notes["eye_blinks"]["confidence"],
            )

        # Create streams for video data.
        if is_stream_video_world:
            self.add_channel(
                bundle_name="eye_video_world",
                channel_name="frame_timestamp",
                data_type="float64",
                sample_size=[1],
                buf_len=buf_len,
                sampling_rate_hz=fps_video_world,
                data_notes=self._data_notes["eye_video_world"]["frame_timestamp"],
            )
            self.add_channel(
                bundle_name="eye_video_world",
                channel_name="frame_index",
                data_type="uint64",
                sample_size=[1],
                buf_len=buf_len,
                sampling_rate_hz=fps_video_world,
                data_notes=self._data_notes["eye_video_world"]["frame_index"],
            )
            self.add_channel(
                bundle_name="eye_video_world",
                channel_name="frame_sequence_id",
                data_type="uint64",
                sample_size=[1],
                buf_len=buf_len,
                sampling_rate_hz=fps_video_world,
                data_notes=self._data_notes["eye_video_world"]["frame_sequence_id"],
            )
            self.add_channel(
                bundle_name="eye_video_world",
                channel_name="frame",
                data_type="uint8",
                sample_size=shape_video_world,
                buf_len=buf_len,
                mem_size=mem_size,
                sampling_rate_hz=fps_video_world,
                data_notes=self._data_notes["eye_video_world"]["frame"],
                is_measure_rate_hz=True,
                is_video=True,
                color_format=self._pixel_format,
                timesteps_before_solidified=self._timesteps_before_solidified,
            )

        if is_stream_video_eye:
            self.add_channel(
                bundle_name="eye_video_eye0",
                channel_name="frame_timestamp",
                data_type="float64",
                sample_size=[1],
                buf_len=buf_len,
                sampling_rate_hz=fps_video_eye0,
                data_notes=self._data_notes["eye_video_eye0"]["frame_timestamp"],
            )
            self.add_channel(
                bundle_name="eye_video_eye0",
                channel_name="frame_index",
                data_type="uint64",
                sample_size=[1],
                buf_len=buf_len,
                sampling_rate_hz=fps_video_eye0,
                data_notes=self._data_notes["eye_video_eye0"]["frame_index"],
            )
            self.add_channel(
                bundle_name="eye_video_eye0",
                channel_name="frame_sequence_id",
                data_type="uint64",
                sample_size=[1],
                buf_len=buf_len,
                sampling_rate_hz=fps_video_eye0,
                data_notes=self._data_notes["eye_video_eye0"]["frame_sequence_id"],
            )
            self.add_channel(
                bundle_name="eye_video_eye0",
                channel_name="frame",
                data_type="uint8",
                sample_size=shape_video_eye0,
                buf_len=buf_len,
                mem_size=mem_size,
                sampling_rate_hz=fps_video_eye0,
                data_notes=self._data_notes["eye_video_eye0"]["frame"],
                is_measure_rate_hz=True,
                is_video=True,
                color_format=self._pixel_format,
                timesteps_before_solidified=self._timesteps_before_solidified,
            )
            if is_binocular:
                self.add_channel(
                    bundle_name="eye_video_eye1",
                    channel_name="frame_timestamp",
                    data_type="float64",
                    sample_size=[1],
                    buf_len=buf_len,
                    sampling_rate_hz=fps_video_eye1,
                    data_notes=self._data_notes["eye_video_eye1"]["frame_timestamp"],
                )
                self.add_channel(
                    bundle_name="eye_video_eye1",
                    channel_name="frame_index",
                    data_type="uint64",
                    sample_size=[1],
                    buf_len=buf_len,
                    sampling_rate_hz=fps_video_eye1,
                    data_notes=self._data_notes["eye_video_eye1"]["frame_index"],
                )
                self.add_channel(
                    bundle_name="eye_video_eye1",
                    channel_name="frame_sequence_id",
                    data_type="uint64",
                    sample_size=[1],
                    buf_len=buf_len,
                    sampling_rate_hz=fps_video_eye1,
                    data_notes=self._data_notes["eye_video_eye1"]["frame_sequence_id"],
                )
                self.add_channel(
                    bundle_name="eye_video_eye1",
                    channel_name="frame",
                    data_type="uint8",
                    sample_size=shape_video_eye1,
                    buf_len=buf_len,
                    mem_size=mem_size,
                    sampling_rate_hz=fps_video_eye1,
                    data_notes=self._data_notes["eye_video_eye1"]["frame"],
                    is_measure_rate_hz=True,
                    is_video=True,
                    color_format=self._pixel_format,
                    timesteps_before_solidified=self._timesteps_before_solidified,
                )

    def get_fps(self) -> dict[str, float | None]:
        fps = {
            "eye_gaze": super()._get_fps("eye_gaze", "toa_s"),
            "eye_pupil": super()._get_fps("eye_pupil", "toa_s"),
            "eye_fixations": super()._get_fps("eye_fixations", "toa_s"),
            "eye_blinks": super()._get_fps("eye_blinks", "toa_s"),
        }

        if self._is_stream_video_world:
            fps["eye_video_world"] = super()._get_fps("eye_video_world", "frame")
        if self._is_stream_video_eye:
            fps["eye_video_eye0"] = super()._get_fps("eye_video_eye0", "frame")
            if self._is_binocular:
                fps["eye_video_eye1"] = super()._get_fps("eye_video_eye1", "frame")
        return fps

    def _define_data_notes(self) -> None:
        self._data_notes = {}
        self._data_notes.setdefault("eye_gaze", {})
        self._data_notes.setdefault("eye_pupil", {})
        self._data_notes.setdefault("eye_fixations", {})
        self._data_notes.setdefault("eye_blinks", {})
        self._data_notes.setdefault("eye_time", {})
        self._data_notes.setdefault("eye_video_eye0", {})
        self._data_notes.setdefault("eye_video_eye1", {})
        self._data_notes.setdefault("eye_video_world", {})

        # Gaze data
        self._data_notes["eye_gaze"]["confidence"] = OrderedDict(
            [
                ("Range", "[0, 1]"),
                ("Description", "Confidence of the gaze detection"),
                ("PupilCapture key", "gaze.Xd. > confidence"),
            ]
        )
        self._data_notes["eye_gaze"]["eye_center_3d"] = OrderedDict(
            [
                ("Units", "mm"),
                (
                    "Notes",
                    "Maps pupil positions into the world camera coordinate system",
                ),
                (DataContainer.metadata_data_headings_key, ["x", "y", "z"]),
                ("PupilCapture key", "gaze.3d. > eye_center_3d"),
            ]
        )
        self._data_notes["eye_gaze"]["normal_3d"] = OrderedDict(
            [
                ("Units", "mm"),
                (
                    "Notes",
                    "Maps pupil positions into the world camera coordinate system",
                ),
                (DataContainer.metadata_data_headings_key, ["x", "y", "z"]),
                ("PupilCapture key", "gaze.3d. > gaze_normal_3d"),
            ]
        )
        self._data_notes["eye_gaze"]["point_3d"] = OrderedDict(
            [
                ("Units", "mm"),
                (
                    "Notes",
                    "Maps pupil positions into the world camera coordinate system",
                ),
                (DataContainer.metadata_data_headings_key, ["x", "y", "z"]),
                ("PupilCapture key", "gaze.3d. > gaze_point_3d"),
            ]
        )
        self._data_notes["eye_gaze"]["position"] = OrderedDict(
            [
                (
                    "Description",
                    "The normalized gaze position in image space, corresponding to the world camera image",
                ),
                ("Units", "normalized between [0, 1]"),
                ("Origin", "bottom left"),
                (DataContainer.metadata_data_headings_key, ["x", "y"]),
                ("PupilCapture key", "gaze.Xd. > norm_pos"),
            ]
        )
        self._data_notes["eye_gaze"]["timestamp"] = OrderedDict(
            [
                (
                    "Description",
                    "The timestamp recorded by the Pupil Capture software, "
                    "which should be more precise than the system time when the data was received (the time_s field). "
                    "Note that Pupil Core time was synchronized with system time at the start of recording, accounting for communication delays.",
                ),
                ("PupilCapture key", "gaze.Xd. > timestamp"),
            ]
        )

        # Pupil data
        self._data_notes["eye_pupil"]["confidence"] = OrderedDict(
            [
                ("Range", "[0, 1]"),
                ("Description", "Confidence of the pupil detection"),
                ("PupilCapture key", "gaze.Xd. > base_data > confidence"),
            ]
        )
        self._data_notes["eye_pupil"]["circle3d_center"] = OrderedDict(
            [
                ("Units", "mm"),
                (DataContainer.metadata_data_headings_key, ["x", "y", "z"]),
                ("PupilCapture key", "gaze.Xd. > base_data > circle_3d > center"),
            ]
        )
        self._data_notes["eye_pupil"]["circle3d_normal"] = OrderedDict(
            [
                ("Units", "mm"),
                (DataContainer.metadata_data_headings_key, ["x", "y", "z"]),
                ("PupilCapture key", "gaze.Xd. > base_data > circle_3d > normal"),
            ]
        )
        self._data_notes["eye_pupil"]["circle3d_radius"] = OrderedDict(
            [
                ("Units", "mm"),
                ("PupilCapture key", "gaze.Xd. > base_data > circle_3d > radius"),
            ]
        )
        self._data_notes["eye_pupil"]["diameter"] = OrderedDict(
            [
                ("Units", "pixels"),
                (
                    "Notes",
                    "The estimated pupil diameter in image space, corresponding to the eye camera image",
                ),
                ("PupilCapture key", "gaze.Xd. > base_data > diameter"),
            ]
        )
        self._data_notes["eye_pupil"]["diameter3d"] = OrderedDict(
            [
                ("Units", "mm"),
                ("Notes", "The estimated pupil diameter in 3D space"),
                ("PupilCapture key", "gaze.Xd. > base_data > diameter_3d"),
            ]
        )
        self._data_notes["eye_pupil"]["polar_phi"] = OrderedDict(
            [
                (
                    "Notes",
                    "Pupil polar coordinate on 3D eye model. The model assumes a fixed eye ball size, so there is no radius key.",
                ),
                ("See also", "polar_theta is the other polar coordinate"),
                ("PupilCapture key", "gaze.Xd. > base_data > phi"),
            ]
        )
        self._data_notes["eye_pupil"]["polar_theta"] = OrderedDict(
            [
                (
                    "Notes",
                    "Pupil polar coordinate on 3D eye model. The model assumes a fixed eye ball size, so there is no radius key.",
                ),
                ("See also", "polar_phi is the other polar coordinate"),
                ("PupilCapture key", "gaze.Xd. > base_data > theta"),
            ]
        )
        self._data_notes["eye_pupil"]["position"] = OrderedDict(
            [
                (
                    "Description",
                    "The normalized pupil position in image space, corresponding to the eye camera image",
                ),
                ("Units", "normalized between [0, 1]"),
                ("Origin", "bottom left"),
                (DataContainer.metadata_data_headings_key, ["x", "y"]),
                ("PupilCapture key", "gaze.Xd. > base_data > norm_pos"),
            ]
        )
        self._data_notes["eye_pupil"]["projected_sphere_angle"] = OrderedDict(
            [
                (
                    "Description",
                    "Projection of the 3D eye ball sphere into image space corresponding to the eye camera image",
                ),
                ("Units", "degrees"),
                ("PupilCapture key", "gaze.Xd. > base_data > projected_sphere > angle"),
            ]
        )
        self._data_notes["eye_pupil"]["projected_sphere_axes"] = OrderedDict(
            [
                (
                    "Description",
                    "Projection of the 3D eye ball sphere into image space corresponding to the eye camera image",
                ),
                ("Units", "pixels"),
                ("Origin", "bottom left"),
                ("PupilCapture key", "gaze.Xd. > base_data > projected_sphere > axes"),
            ]
        )
        self._data_notes["eye_pupil"]["projected_sphere_center"] = OrderedDict(
            [
                (
                    "Description",
                    "Projection of the 3D eye ball sphere into image space corresponding to the eye camera image",
                ),
                ("Units", "pixels"),
                ("Origin", "bottom left"),
                (DataContainer.metadata_data_headings_key, ["x", "y"]),
                (
                    "PupilCapture key",
                    "gaze.Xd. > base_data > projected_sphere > center",
                ),
            ]
        )
        self._data_notes["eye_pupil"]["sphere_center"] = OrderedDict(
            [
                ("Description", "The 3D eye ball sphere"),
                ("Units", "mm"),
                (DataContainer.metadata_data_headings_key, ["x", "y", "z"]),
                ("PupilCapture key", "gaze.Xd. > base_data > sphere > center"),
            ]
        )
        self._data_notes["eye_pupil"]["sphere_radius"] = OrderedDict(
            [
                ("Description", "The 3D eye ball sphere"),
                ("Units", "mm"),
                ("PupilCapture key", "gaze.Xd. > base_data > sphere > radius"),
            ]
        )
        self._data_notes["eye_pupil"]["timestamp"] = OrderedDict(
            [
                (
                    "Description",
                    "The timestamp recorded by the Pupil Capture software, "
                    "which should be more precise than the system time when the data was received (the time_s field). "
                    "Note that Pupil Core time was synchronized with system time at the start of recording, accounting for communication delays.",
                ),
                ("PupilCapture key", "gaze.Xd. > base_data > timestamp"),
            ]
        )

        # Fixations data.
        self._data_notes["eye_fixations"]["id"] = OrderedDict(
            [
                (
                    "Description",
                    "The index of the fixation which relates it to other data.",
                ),
                ("PupilCapture key", "fixations. > id"),
            ]
        )
        self._data_notes["eye_fixations"]["timestamp"] = OrderedDict(
            [
                (
                    "Description",
                    "The timestamp recorded by the Pupil Capture software, "
                    "which should be more precise than the system time when the data was received (the time_s field). "
                    "Note that Pupil Core time was synchronized with system time at the start of recording, accounting for communication delays.",
                ),
                ("PupilCapture key", "fixations. > timestamp"),
            ]
        )
        self._data_notes["eye_fixations"]["norm_pos"] = OrderedDict(
            [
                (
                    "Description",
                    "The normalized pupil position in image space, corresponding to the eye camera image",
                ),
                ("Units", "normalized between [0, 1]"),
                ("Origin", "bottom left"),
                (DataContainer.metadata_data_headings_key, ["x", "y"]),
                ("PupilCapture key", "fixations. > norm_pos"),
            ]
        )
        self._data_notes["eye_fixations"]["dispersion"] = OrderedDict(
            [
                ("Range", ""),
                ("Description", ""),
                ("PupilCapture key", "fixations. > dispersion"),
            ]
        )
        self._data_notes["eye_fixations"]["duration"] = OrderedDict(
            [
                ("Range", ""),
                ("Description", ""),
                ("PupilCapture key", "fixations. > duration"),
            ]
        )
        self._data_notes["eye_fixations"]["confidence"] = OrderedDict(
            [
                ("Range", ""),
                ("Description", ""),
                ("PupilCapture key", "fixations. > confidence"),
            ]
        )
        self._data_notes["eye_fixations"]["gaze_point_3d"] = OrderedDict(
            [
                ("Units", "mm"),
                (
                    "Notes",
                    "Maps pupil positions into the world camera coordinate system",
                ),
                (DataContainer.metadata_data_headings_key, ["x", "y", "z"]),
                ("PupilCapture key", "fixations. > gaze_point_3d"),
            ]
        )

        # Blinks data
        self._data_notes["eye_blinks"]["confidence"] = OrderedDict(
            [
                ("Range", "[0, 1]"),
                ("Description", "Confidence of the blink detection"),
                ("PupilCapture key", "blinks. > confidence"),
            ]
        )
        self._data_notes["eye_blinks"]["timestamp"] = OrderedDict(
            [
                (
                    "Description",
                    "The timestamp recorded by the Pupil Capture software, "
                    "which should be more precise than the system time when the data was received (the time_s field). "
                    "Note that Pupil Core time was synchronized with system time at the start of recording, accounting for communication delays.",
                ),
                ("PupilCapture key", "blinks. > timestamp"),
            ]
        )

        # Time
        self._data_notes["eye_time"]["device_time_s"] = OrderedDict(
            [
                (
                    "Description",
                    "The timestamp fetched from the Pupil Core service, which can be used for alignment to system time in time_s. "
                    "As soon as system time time_s was recorded, a command was sent to Pupil Capture to get its time; "
                    "so a slight communication delay is included on the order of milliseconds.  "
                    "Note that Pupil Core time was synchronized with system time at the start of recording, accounting for communication delays.",
                ),
            ]
        )

        # Eye videos
        for i in range(2):
            self._data_notes["eye_video_eye%s" % i]["frame_timestamp"] = OrderedDict(
                [
                    (
                        "Description",
                        "The timestamp recorded by the Pupil Core service, "
                        "which should be more precise than the system time when the data was received (the time_s field). "
                        "Note that Pupil Core time was synchronized with system time at the start of recording, accounting for communication delays.",
                    ),
                ]
            )
            self._data_notes["eye_video_eye%s" % i]["frame_index"] = OrderedDict(
                [
                    (
                        "Description",
                        "The frame index starting from 0, w.r.t. to that of the Pupil Core service.",
                    ),
                ]
            )
            self._data_notes["eye_video_eye%s" % i]["frame_sequence_id"] = OrderedDict(
                [
                    (
                        "Description",
                        "The frame index recorded by the Pupil Core service, "
                        "which relates to world frame used for annotation",
                    ),
                ]
            )
            self._data_notes["eye_video_eye%s" % i]["frame"] = OrderedDict(
                [
                    ("Format", "Frames are in BGR format"),
                ]
            )
        # World video
        self._data_notes["eye_video_world"]["frame_timestamp"] = OrderedDict(
            [
                (
                    "Description",
                    "The timestamp recorded by the Pupil Core service, "
                    "which should be more precise than the system time when the data was received (the time_s field). "
                    "Note that Pupil Core time was synchronized with system time at the start of recording, accounting for communication delays.",
                ),
            ]
        )
        self._data_notes["eye_video_world"]["frame_index"] = OrderedDict(
            [
                (
                    "Description",
                    "The frame index starting from 0, w.r.t. to that of the Pupil Core service.",
                ),
            ]
        )
        self._data_notes["eye_video_world"]["frame_sequence_id"] = OrderedDict(
            [
                (
                    "Description",
                    "The frame index recorded by the Pupil Core service, "
                    "which relates to world frame used for annotation",
                ),
            ]
        )
        self._data_notes["eye_video_world"]["frame"] = OrderedDict(
            [
                ("Format", "Frames are in BGR format"),
            ]
        )
