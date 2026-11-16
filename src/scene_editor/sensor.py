"""The sensor line and the frames it records.

A line of reflective IR distance sensors is polled at a fixed rate while objects pass
under it. Successive polls stacked together give a space-time picture: one axis holds
the channels (sensors of the line), the other the polls (time). A frame is a window of
`frame_length` consecutive polls; a tape is a recording of any length.

Inside the program every object and tape is *time-major*: rows are polls, columns are
channels. Stored frames may hold the channels on either image axis (`sensor_axis`),
so a frame is read and written in its own orientation and turned where needed.
"""

from dataclasses import dataclass


SENSOR_AXES = ("x", "y")
# Sanity bounds of a frame geometry, far above any line in use (the reference line has 40 channels).
MAX_CHANNELS = 1024
MAX_FRAME_LENGTH = 4096


@dataclass(frozen=True)
class FrameGeometry:
    """Geometry of the sensor line and of its frames."""

    channels: int        # sensors in the line
    frame_length: int    # polls in one frame
    sensor_axis: str = "x"  # image axis of the channels in stored frames

    def __post_init__(self):
        if self.sensor_axis not in SENSOR_AXES:
            raise ValueError("sensor_axis must be x or y")
        for name, value, limit in (("channels", self.channels, MAX_CHANNELS),
                                   ("frame_length", self.frame_length, MAX_FRAME_LENGTH)):
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= limit:
                raise ValueError(f"{name} must be an integer in [1, {limit}]")

    @property
    def frame_shape(self):
        """(rows, columns) of a stored frame, in its own orientation."""
        if self.sensor_axis == "x":
            return (self.frame_length, self.channels)
        return (self.channels, self.frame_length)

    @property
    def time_major_shape(self):
        """(rows, columns) of a frame turned so rows are polls and columns are channels."""
        return (self.frame_length, self.channels)

    @classmethod
    def from_frame(cls, shape, sensor_axis):
        """The geometry of frames of `shape` (rows, columns) with the channels on `sensor_axis`."""
        rows, columns = shape
        if sensor_axis == "x":
            return cls(int(columns), int(rows), sensor_axis)
        return cls(int(rows), int(columns), sensor_axis)

    def payload(self):
        rows, columns = self.frame_shape
        return dict(channels=self.channels, frame_length=self.frame_length, sensor_axis=self.sensor_axis,
                    frame_width=columns, frame_height=rows)
