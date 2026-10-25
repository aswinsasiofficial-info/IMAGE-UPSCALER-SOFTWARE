"""Validation of the Upscaler's settings (Django forms, used by the desktop UI).

This is where Django still fits the desktop-only app: the screen collects
its widget values into a dict and this form checks them in one place —
allowed factors, output folder, JPEG quality, and that no output would exceed
the format's size limit. Django is configured standalone here (no web
server, no database, no apps).
"""

from __future__ import annotations

from pathlib import Path

import django
from django.conf import settings

if not settings.configured:
    settings.configure(USE_I18N=False, USE_TZ=True)
    django.setup()

from django import forms  # noqa: E402  (needs the settings above)

from .files import DEFAULT_OUTPUT_DIR  # noqa: E402
from .geometry import DEFAULT_FACTOR, FACTORS, MAX_SIDE, plan_size  # noqa: E402
from .workspace import JobOptions  # noqa: E402


class UpscaleForm(forms.Form):
    factor = forms.TypedChoiceField(label="Upscale factor", coerce=int,
                                    choices=[(f, f"{f}×") for f in FACTORS],
                                    initial=DEFAULT_FACTOR)
    output_dir = forms.CharField(label="Output folder", required=False, max_length=1024)
    save_beside = forms.BooleanField(required=False)
    device = forms.ChoiceField(label="Device",
                               choices=[("auto", "Auto"), ("cuda", "CUDA (GPU)"), ("cpu", "CPU")])
    fp16 = forms.BooleanField(required=False)
    tile = forms.IntegerField(label="Tile size", min_value=0, max_value=1024)
    output_format = forms.ChoiceField(label="Format", choices=[("jpg", "JPEG"), ("png", "PNG")],
                                      initial="jpg")
    jpg_quality = forms.IntegerField(label="JPEG quality", required=False, min_value=50,
                                     max_value=100)
    rename_batch = forms.BooleanField(required=False, initial=True)

    def __init__(self, *args, images: list[tuple[int, int]] = (), **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.images = list(images)      # upright (width, height) of every queued image

    def clean_output_dir(self) -> str:
        value = (self.cleaned_data.get("output_dir") or "").strip().strip('"')
        if value and not self.data.get("save_beside") and Path(value).expanduser().is_file():
            raise forms.ValidationError("That path is a file, not a folder.")
        return value

    def clean(self) -> dict:
        data = super().clean()
        if data.get("output_format") == "jpg" and data.get("jpg_quality") is None \
                and "jpg_quality" not in self.errors:
            self.add_error("jpg_quality", "Enter a JPEG quality from 50 to 100.")
        factor = data.get("factor")
        if factor in FACTORS and self.images:
            too_big = sum(max(plan_size(w, h, factor).output) > MAX_SIDE for w, h in self.images)
            if too_big:
                self.add_error("factor", (
                    f"{too_big} image(s) would be larger than {MAX_SIDE:,} px on a side at "
                    f"{factor}×. Choose a smaller factor."))
        return data

    def job_options(self) -> JobOptions:
        data = self.cleaned_data
        output_dir = data.get("output_dir")
        return JobOptions(
            factor=data["factor"],
            output_dir=Path(output_dir).expanduser() if output_dir else DEFAULT_OUTPUT_DIR,
            save_beside=data.get("save_beside", False),
            device=data["device"],
            fp16=data.get("fp16", False),
            tile=data["tile"],
            output_format=data["output_format"],
            jpg_quality=data.get("jpg_quality") or 95,
            rename_batch=data.get("rename_batch", False),
        )

    def first_error(self) -> str:
        """``"Label: message"`` for the first problem, for a message box."""
        for field, messages in self.errors.items():
            label = self.fields[field].label if field in self.fields else None
            return f"{label}: {messages[0]}" if label else messages[0]
        return ""
