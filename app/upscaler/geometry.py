"""Output-size planning for the Upscaler (pure arithmetic, no image I/O).

Every image keeps its full composition and aspect ratio by default:

  * ``scale`` mode multiplies width *and* height by one factor, so a
    1024 × 1536 portrait at 4× becomes exactly 4096 × 6144.
  * ``exact`` mode targets a fixed W × H frame. When the frame has the image's
    own shape the image is simply resized into it. When the shapes differ the
    image cannot fill the frame without gaining borders or losing its edges,
    so the plan stays *unresolved* until the user explicitly picks ``pad`` or
    ``crop``. There is no stretch option: content is never distorted.

The same plan drives the preview shown before processing and the pixels
written afterwards, so what the user is shown is what gets saved.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

SCALE, EXACT = "scale", "exact"
PAD, CROP = "pad", "crop"

MODEL_SCALE = 4            # Real-ESRGAN x4plus always enlarges exactly 4x
MIN_FACTOR, MAX_FACTOR = 1, 8
DEFAULT_FACTOR = 4
MAX_SIDE = 65500           # libjpeg's hard limit, and a sane memory ceiling for PNG
LARGE_OUTPUT_MP = 150      # above this the preview warns about RAM use


def round_half_up(value: float) -> int:
    return int(math.floor(value + 0.5))


def format_factor(factor: float) -> str:
    """4.0 -> '4', 2.5 -> '2.5'."""
    return f"{factor:.2f}".rstrip("0").rstrip(".")


def ratio_label(width: int, height: int) -> str:
    """Human aspect ratio: '2:3', '16:9', or '1.78:1' when it doesn't reduce nicely."""
    g = math.gcd(width, height)
    a, b = width // g, height // g
    if a <= 32 and b <= 32:
        return f"{a}:{b}"
    return f"{width / height:.2f}:1" if width >= height else f"1:{height / width:.2f}"


def orientation(width: int, height: int) -> str:
    if width > height:
        return "landscape"
    return "portrait" if height > width else "square"


@dataclass(frozen=True)
class SizeOptions:
    mode: str = SCALE
    factor: float = DEFAULT_FACTOR
    width: int = 0          # exact mode frame
    height: int = 0
    fit: str = ""           # "", PAD or CROP — only consulted when shapes differ


@dataclass(frozen=True)
class Plan:
    """Where one image ends up.

    ``content`` is the size the *whole* picture is resized to (one factor for
    both axes). ``offset`` is where its top-left corner lands on the output
    canvas: positive means borders were added, negative means the edges
    hang outside the frame and are trimmed.
    """

    source: tuple[int, int]
    output: tuple[int, int]
    content: tuple[int, int]
    offset: tuple[int, int]
    mode: str
    fit: str = ""
    conflict: bool = False      # exact frame has a different shape from the image

    @property
    def resolved(self) -> bool:
        return not self.conflict or self.fit in (PAD, CROP)

    @property
    def padding(self) -> tuple[int, int, int, int]:
        """Border widths (left, top, right, bottom) added around the picture."""
        if self.fit != PAD:
            return (0, 0, 0, 0)
        (ow, oh), (cw, ch), (x, y) = self.output, self.content, self.offset
        return (x, y, ow - cw - x, oh - ch - y)

    @property
    def cropped(self) -> tuple[int, int, int, int]:
        """Pixels (left, top, right, bottom) trimmed from the resized picture."""
        if self.fit != CROP:
            return (0, 0, 0, 0)
        (ow, oh), (cw, ch), (x, y) = self.output, self.content, self.offset
        return (-x, -y, cw - ow + x, ch - oh + y)

    @property
    def lost_fraction(self) -> float:
        if self.fit != CROP:
            return 0.0
        (ow, oh), (cw, ch) = self.output, self.content
        return 1.0 - (ow * oh) / (cw * ch)

    @property
    def megapixels(self) -> float:
        return self.output[0] * self.output[1] / 1_000_000


def same_shape(width: int, height: int, frame_w: int, frame_h: int) -> bool:
    """True if one factor s rounds the image to exactly the frame on both axes.

    That's the same pixel rounding scale mode uses, so a frame typed from an
    image's own proportions (e.g. 1920x1080 -> 5005x2815) counts as a match.
    """
    lo = max((frame_w - 0.5) / width, (frame_h - 0.5) / height)
    hi = min((frame_w + 0.5) / width, (frame_h + 0.5) / height)
    return lo < hi


def plan_size(width: int, height: int, opts: SizeOptions) -> Plan:
    """Plan the output for a ``width`` × ``height`` (EXIF-upright) image."""
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    source = (width, height)

    if opts.mode == SCALE:
        size = (max(1, round_half_up(width * opts.factor)),
                max(1, round_half_up(height * opts.factor)))
        return Plan(source, size, size, (0, 0), SCALE)
    if opts.mode != EXACT:
        raise ValueError(f"unknown size mode {opts.mode!r}")

    frame = (opts.width, opts.height)
    if frame[0] <= 0 or frame[1] <= 0:
        raise ValueError("exact size needs a positive width and height")
    if same_shape(width, height, *frame):
        return Plan(source, frame, frame, (0, 0), EXACT)
    # Largest uniform resize that fits entirely inside the frame.
    fit_scale = min(frame[0] / width, frame[1] / height)
    fitted = (min(frame[0], max(1, round_half_up(width * fit_scale))),
              min(frame[1], max(1, round_half_up(height * fit_scale))))

    if opts.fit == CROP:
        # Smallest uniform resize that covers the frame; centred trim of the rest.
        fill_scale = max(frame[0] / width, frame[1] / height)
        filled = (max(frame[0], round_half_up(width * fill_scale)),
                  max(frame[1], round_half_up(height * fill_scale)))
        offset = (-((filled[0] - frame[0]) // 2), -((filled[1] - frame[1]) // 2))
        return Plan(source, frame, filled, offset, EXACT, CROP, conflict=True)

    # PAD, or not chosen yet (previewed as pad but left unresolved).
    offset = ((frame[0] - fitted[0]) // 2, (frame[1] - fitted[1]) // 2)
    return Plan(source, frame, fitted, offset, EXACT,
                PAD if opts.fit == PAD else "", conflict=True)


def _edges(left: int, top: int, right: int, bottom: int) -> str:
    parts = []
    if left or right:
        parts.append(f"{left} px left + {right} px right" if left != right
                     else f"{left} px left and right")
    if top or bottom:
        parts.append(f"{top} px top + {bottom} px bottom" if top != bottom
                     else f"{top} px top and bottom")
    return ", ".join(parts)


def describe(plan: Plan, opts: SizeOptions) -> str:
    """One-line explanation of a plan, shown in the queue before processing."""
    sw, sh = plan.source
    shape = ratio_label(sw, sh)
    if plan.mode == SCALE:
        return (f"Scaled {format_factor(opts.factor)}× on both sides — "
                f"same {shape} shape, whole picture kept")
    if not plan.conflict:
        return f"Resized to fit the frame exactly — same {shape} shape, whole picture kept"
    frame_shape = ratio_label(*plan.output)
    if plan.fit == PAD:
        cw, ch = plan.content
        return (f"Whole {shape} picture fitted to {cw} × {ch}, "
                f"borders {_edges(*plan.padding)}")
    if plan.fit == CROP:
        cw, ch = plan.content
        return (f"Filled to {cw} × {ch}, trimmed {_edges(*plan.cropped)} "
                f"({plan.lost_fraction:.0%} of the picture lost)")
    return f"{shape} picture vs {frame_shape} frame — choose Pad or Crop"
