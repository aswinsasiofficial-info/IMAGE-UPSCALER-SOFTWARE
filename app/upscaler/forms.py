"""Validation of the Upscaler's settings.

Besides plain field checks, the form enforces the shape rule: in exact-size
mode, if any queued image has a different aspect ratio from the requested
frame, the user must explicitly choose Pad or Crop before a batch can start.
There is no default for that choice and no stretch option.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from django import forms

from .files import DEFAULT_OUTPUT_DIR, INVALID_NAME_CHARS
from .geometry import (CROP, DEFAULT_FACTOR, EXACT, MAX_FACTOR, MAX_SIDE, MIN_FACTOR, PAD,
                       SCALE, SizeOptions, plan_size)
from .imaging import PAD_COLORS
from .workspace import JobOptions


class UpscaleForm(forms.Form):
    size_mode = forms.ChoiceField(label="Output size",
                                  choices=[(SCALE, "Scale by"), (EXACT, "Exact size")])
    scale_factor = forms.DecimalField(label="Scale factor", required=False, min_value=MIN_FACTOR,
                                      max_value=MAX_FACTOR, max_digits=4, decimal_places=2,
                                      initial=DEFAULT_FACTOR)
    exact_width = forms.IntegerField(label="Exact width", required=False, min_value=1,
                                     max_value=MAX_SIDE)
    exact_height = forms.IntegerField(label="Exact height", required=False, min_value=1,
                                      max_value=MAX_SIDE)
    fit = forms.ChoiceField(label="Pad or crop", required=False,
                            choices=[("", "Not chosen"), (PAD, "Pad to fit"), (CROP, "Crop to fill")])
    pad_color = forms.ChoiceField(label="Border colour", required=False,
                                  choices=[(c, c) for c in PAD_COLORS])
    output_dir = forms.CharField(label="Output folder", required=False, max_length=1024)
    save_beside = forms.BooleanField(required=False)
    device = forms.ChoiceField(label="Device",
                               choices=[("auto", "Auto"), ("cuda", "CUDA (GPU)"), ("cpu", "CPU")])
    fp16 = forms.BooleanField(required=False)
    tile = forms.IntegerField(label="Tile size", min_value=0, max_value=1024)
    output_format = forms.ChoiceField(label="Format", choices=[("png", "png"), ("jpg", "jpg")])
    jpg_quality = forms.IntegerField(label="JPG quality", min_value=50, max_value=100)
    suffix = forms.CharField(label="Filename suffix", required=False, max_length=64, strip=False)
    rename_batch = forms.BooleanField(required=False)

    def __init__(self, *args, images: list[tuple[int, int]] = (), **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.images = list(images)      # upright (width, height) of every queued image
        # Settings the page greys out for the current choices aren't validated,
        # so a stale value in a disabled box can never block a batch.
        raw = self.data
        flag = forms.BooleanField(required=False).to_python
        unused = []
        if raw.get("size_mode") == EXACT:
            unused.append("scale_factor")
        elif raw.get("size_mode") == SCALE:
            unused += ["exact_width", "exact_height", "fit", "pad_color"]
        if raw.get("output_format") != "jpg":
            unused.append("jpg_quality")
        if flag(raw.get("rename_batch")):
            unused.append("suffix")
        if flag(raw.get("save_beside")):
            unused.append("output_dir")
        for name in unused:
            self.fields.pop(name, None)

    def clean_suffix(self) -> str:
        suffix = self.cleaned_data.get("suffix") or ""
        bad = sorted(set(suffix) & INVALID_NAME_CHARS)
        if bad:
            raise forms.ValidationError(
                f"The suffix can't contain {' '.join(bad)} — Windows doesn't allow them in file names.")
        return suffix

    def clean_output_dir(self) -> str:
        value = (self.cleaned_data.get("output_dir") or "").strip().strip('"')
        if value and Path(value).expanduser().is_file():
            raise forms.ValidationError("That path is a file, not a folder.")
        return value

    def clean(self) -> dict:
        data = super().clean()
        mode = data.get("size_mode")
        if mode == SCALE and data.get("scale_factor") is None and "scale_factor" not in self.errors:
            self.add_error("scale_factor", f"Enter a scale factor from {MIN_FACTOR} to {MAX_FACTOR}.")
        if mode == EXACT:
            for name in ("exact_width", "exact_height"):
                if data.get(name) is None and name not in self.errors:
                    self.add_error(name, "Enter the exact output size in pixels.")

        opts = self.size_options()
        if opts is not None and self.images:
            plans = [plan_size(w, h, opts) for w, h in self.images]
            conflicts = sum(p.conflict for p in plans)
            if conflicts and data.get("fit") not in (PAD, CROP):
                self.add_error("fit", (
                    f"{conflicts} of {len(plans)} image(s) have a different shape from "
                    f"{opts.width} × {opts.height}. Choose Pad (keep the whole picture, add "
                    f"borders) or Crop (fill the frame, trim the edges) — nothing is cropped "
                    f"or stretched without your say-so."))
            too_big = sum(max(p.output) > MAX_SIDE for p in plans)
            if too_big:
                # Not tied to the size field: that would drop it from cleaned_data and
                # blank every row of the preview instead of flagging the big ones.
                self.add_error(None, (
                    f"{too_big} image(s) would be larger than {MAX_SIDE:,} px on a side. "
                    f"Use a smaller {'factor' if mode == SCALE else 'size'}."))

        if (mode == EXACT and data.get("fit") == PAD and data.get("pad_color") == "transparent"
                and data.get("output_format") == "jpg"):
            self.add_error("pad_color", "JPEG can't store transparency — pick White or Black "
                                        "borders, or save as png.")
        return data

    def size_options(self) -> Optional[SizeOptions]:
        """The size settings, or None while they're incomplete/invalid."""
        data = getattr(self, "cleaned_data", {})
        mode = data.get("size_mode")
        if mode == SCALE and data.get("scale_factor") is not None:
            return SizeOptions(SCALE, factor=float(data["scale_factor"]))
        if mode == EXACT and data.get("exact_width") and data.get("exact_height"):
            return SizeOptions(EXACT, width=data["exact_width"], height=data["exact_height"],
                               fit=data.get("fit") or "")
        return None

    def job_options(self) -> JobOptions:
        data = self.cleaned_data
        output_dir = data.get("output_dir")
        return JobOptions(
            size=self.size_options(),
            pad_color=data.get("pad_color") or "white",
            output_dir=Path(output_dir).expanduser() if output_dir else DEFAULT_OUTPUT_DIR,
            save_beside=data.get("save_beside", False),
            device=data["device"],
            fp16=data.get("fp16", False),
            tile=data["tile"],
            output_format=data["output_format"],
            jpg_quality=data.get("jpg_quality") or 95,     # absent when saving png
            suffix=data.get("suffix", ""),
            rename_batch=data.get("rename_batch", False),
        )
