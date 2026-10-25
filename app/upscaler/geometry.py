"""Output-size planning for the Upscaler (pure arithmetic, no image I/O).

Every image keeps its full composition and aspect ratio: width and height are
multiplied by the same whole-number factor (1×–5×), so a 1024 × 1536 portrait
at 4× becomes exactly 4096 × 6144 and at 1× stays 1024 × 1536. Nothing is
cropped, padded, stretched or zoomed.

Real-ESRGAN x4plus has exactly one native scale — 4× (its network ends in two
fixed 2× upsampling stages). Every other factor starts from that 4× result and
is finished with one Lanczos resize using the same factor on both axes;
:func:`how_produced` says so in plain words for the UI.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

MODEL_SCALE = 4                    # the only scale the network produces
FACTORS = (1, 2, 3, 4, 5)
DEFAULT_FACTOR = 4
MAX_SIDE = 65500                   # libjpeg's hard limit, and a sane memory ceiling for PNG
LARGE_OUTPUT_MP = 150              # above this the UI warns about RAM use

DETAIL_NOTE = ("More pixels don't automatically mean more real detail: the AI can only "
               "reconstruct plausible detail, not recover what the original never captured.")


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


def how_produced(factor: int) -> str:
    """How the model gets to *factor* — never claims more than it does natively."""
    if factor == MODEL_SCALE:
        return "Native: Real-ESRGAN x4plus enlarges exactly 4× on both sides."
    if factor == 1:
        return ("Same size. The model runs at its native 4×, then the result is reduced back "
                "to the original size with a Lanczos filter — a clean-up, not more pixels.")
    if factor < MODEL_SCALE:
        return (f"The model runs at its native 4×, then the result is reduced to {factor}× "
                "with a high-quality Lanczos filter.")
    return (f"The model runs at its native 4×; the remaining step to {factor}× is ordinary "
            "Lanczos interpolation — it adds pixels, not new detail.")


@dataclass(frozen=True)
class Plan:
    source: tuple[int, int]        # upright (EXIF-applied) size
    output: tuple[int, int]
    factor: int

    @property
    def megapixels(self) -> float:
        return self.output[0] * self.output[1] / 1_000_000

    @property
    def peak_megapixels(self) -> float:
        """Largest image held in memory while processing (the 4× result or the output)."""
        w, h = self.source
        return max(self.megapixels, w * h * MODEL_SCALE ** 2 / 1_000_000)


def plan_size(width: int, height: int, factor: int) -> Plan:
    """Output size of a ``width`` × ``height`` image at ``factor`` (same on both sides)."""
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    if factor not in FACTORS:
        raise ValueError(f"factor must be one of {FACTORS}")
    return Plan((width, height), (width * factor, height * factor), factor)


def describe(plan: Plan) -> str:
    """One-line explanation shown next to each image before processing."""
    shape = ratio_label(*plan.source)
    if plan.factor == 1:
        return f"Same size — {shape} shape, whole picture kept"
    return f"{plan.factor}× on both sides — same {shape} shape, whole picture kept"
