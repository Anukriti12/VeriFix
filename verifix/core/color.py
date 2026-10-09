"""
core/color.py - Color utilities shared by the renderer, the objective, and the perturbations.

  to_hex / rgb        : normalize any color (hex string, RGB list/tuple) to "#rrggbb" / (r,g,b)
  contrast_ratio      : WCAG 2.1 contrast ratio between two colors (1..21)
  ciede2000           : CIEDE2000 color difference (Sharma, Wu, Dalal 2005), used to calibrate
                        the palette perturbation exactly as Perturb & Invert does (Table 8)
  shift_hue_sat       : rotate hue (degrees) and scale saturation of a color in HSV
  lerp                : linear interpolation between two colors (P&I contrast degradation)

No model or rendering dependency.
"""
from __future__ import annotations

import colorsys
import math
from typing import Any, Tuple


def rgb(color: Any) -> Tuple[int, int, int]:
    """Coerce a color (hex string or RGB list/tuple) to an (r,g,b) int tuple."""
    if isinstance(color, (list, tuple)) and len(color) >= 3:
        return (int(color[0]), int(color[1]), int(color[2]))
    s = str(color).strip().lstrip("#")
    if len(s) == 3:
        s = "".join(c * 2 for c in s)
    try:
        return tuple(int(s[i:i + 2], 16) for i in (0, 2, 4))  # type: ignore
    except Exception:
        return (0, 0, 0)


def to_hex(color: Any) -> str:
    r, g, b = rgb(color)
    return "#{:02x}{:02x}{:02x}".format(max(0, min(255, r)), max(0, min(255, g)),
                                        max(0, min(255, b)))


# ---------------------------------------------------------------- WCAG contrast
def _lin(c: float) -> float:
    c = c / 255.0
    return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4


def luminance(color: Any) -> float:
    r, g, b = rgb(color)
    return 0.2126 * _lin(r) + 0.7152 * _lin(g) + 0.0722 * _lin(b)


def contrast_ratio(c1: Any, c2: Any) -> float:
    l1, l2 = luminance(c1), luminance(c2)
    hi, lo = max(l1, l2), min(l1, l2)
    return (hi + 0.05) / (lo + 0.05)


# ---------------------------------------------------------------- interpolation
def lerp(c1: Any, c2: Any, t: float) -> str:
    """Linear interpolation in RGB: t=0 -> c1, t=1 -> c2."""
    a, b = rgb(c1), rgb(c2)
    return to_hex(tuple(round(a[i] + (b[i] - a[i]) * t) for i in range(3)))


# ---------------------------------------------------------------- HSV shift
def shift_hue_sat(color: Any, hue_deg: float, sat_scale: float = 1.0) -> str:
    r, g, b = (x / 255.0 for x in rgb(color))
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    h = (h + hue_deg / 360.0) % 1.0
    s = max(0.0, min(1.0, s * sat_scale))
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return to_hex((round(r * 255), round(g * 255), round(b * 255)))


# ---------------------------------------------------------------- CIEDE2000
def _srgb_to_lab(color: Any) -> Tuple[float, float, float]:
    r, g, b = rgb(color)

    def f(c):
        c = c / 255.0
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    R, G, B = f(r), f(g), f(b)
    X = (0.4124564 * R + 0.3575761 * G + 0.1804375 * B) / 0.95047
    Y = (0.2126729 * R + 0.7151522 * G + 0.0721750 * B) / 1.00000
    Z = (0.0193339 * R + 0.1191920 * G + 0.9503041 * B) / 1.08883

    def g_(t):
        return t ** (1 / 3) if t > 216 / 24389 else (24389 / 27 * t + 16) / 116

    fx, fy, fz = g_(X), g_(Y), g_(Z)
    return 116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)


def ciede2000_lab(lab1, lab2, kL: float = 1.0, kC: float = 1.0, kH: float = 1.0) -> float:
    """CIEDE2000 difference between two CIELAB colors (Sharma, Wu, Dalal 2005)."""
    L1, a1, b1 = lab1
    L2, a2, b2 = lab2
    C1, C2 = math.hypot(a1, b1), math.hypot(a2, b2)
    Cbar = (C1 + C2) / 2
    G = 0.5 * (1 - math.sqrt(Cbar ** 7 / (Cbar ** 7 + 25 ** 7)))
    a1p, a2p = (1 + G) * a1, (1 + G) * a2
    C1p, C2p = math.hypot(a1p, b1), math.hypot(a2p, b2)
    h1p = math.degrees(math.atan2(b1, a1p)) % 360 if (a1p or b1) else 0.0
    h2p = math.degrees(math.atan2(b2, a2p)) % 360 if (a2p or b2) else 0.0
    dLp = L2 - L1
    dCp = C2p - C1p
    if C1p * C2p == 0:
        dhp = 0.0
    else:
        dhp = h2p - h1p
        if dhp > 180:
            dhp -= 360
        elif dhp < -180:
            dhp += 360
    dHp = 2 * math.sqrt(C1p * C2p) * math.sin(math.radians(dhp) / 2)
    Lbp = (L1 + L2) / 2
    Cbp = (C1p + C2p) / 2
    if C1p * C2p == 0:
        hbp = h1p + h2p
    elif abs(h1p - h2p) <= 180:
        hbp = (h1p + h2p) / 2
    elif h1p + h2p < 360:
        hbp = (h1p + h2p + 360) / 2
    else:
        hbp = (h1p + h2p - 360) / 2
    T = (1 - 0.17 * math.cos(math.radians(hbp - 30)) + 0.24 * math.cos(math.radians(2 * hbp))
         + 0.32 * math.cos(math.radians(3 * hbp + 6)) - 0.20 * math.cos(math.radians(4 * hbp - 63)))
    dtheta = 30 * math.exp(-(((hbp - 275) / 25) ** 2))
    Rc = 2 * math.sqrt(Cbp ** 7 / (Cbp ** 7 + 25 ** 7))
    Sl = 1 + (0.015 * (Lbp - 50) ** 2) / math.sqrt(20 + (Lbp - 50) ** 2)
    Sc = 1 + 0.045 * Cbp
    Sh = 1 + 0.015 * Cbp * T
    Rt = -math.sin(math.radians(2 * dtheta)) * Rc
    return math.sqrt((dLp / (kL * Sl)) ** 2 + (dCp / (kC * Sc)) ** 2 + (dHp / (kH * Sh)) ** 2
                     + Rt * (dCp / (kC * Sc)) * (dHp / (kH * Sh)))


def ciede2000(c1: Any, c2: Any) -> float:
    return ciede2000_lab(_srgb_to_lab(c1), _srgb_to_lab(c2))


def is_hex_color(v: Any) -> bool:
    return isinstance(v, str) and v.strip().startswith("#") and len(v.strip()) in (4, 7)


# ---------------------------------------------------------------- parsing (dataset colors)
def parse_color(value: Any):
    """Parse a color as datasets store it: "#rgb", "#rrggbb", "#rrggbbaa", "rgb(r,g,b)",
    "rgba(r,g,b,a)", or a list/tuple of 3-4 numbers (0-255, or 0-1 floats).
    Returns ("#rrggbb", alpha in [0,1]) or None when the value is not a color."""
    import re
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        if len(value) < 3:
            return None
        try:
            nums = [float(v) for v in value[:4]]
        except Exception:
            return None
        if all(0.0 <= v <= 1.0 for v in nums[:3]) and any(0.0 < v < 1.0 for v in nums[:3]):
            nums = [v * 255.0 for v in nums[:3]] + nums[3:]
        a = nums[3] if len(nums) > 3 else 1.0
        if a > 1.0:
            a = a / 255.0
        return to_hex([int(v + 0.5) for v in nums[:3]]), max(0.0, min(1.0, a))
    s = str(value).strip().lower()
    if not s:
        return None
    m = re.fullmatch(r"#?([0-9a-f]{3}|[0-9a-f]{6}|[0-9a-f]{8})", s)
    if m:
        h = m.group(1)
        if len(h) == 3:
            h = "".join(c * 2 for c in h)
        a = int(h[6:8], 16) / 255.0 if len(h) == 8 else 1.0
        return "#" + h[:6], a
    m = re.fullmatch(r"rgba?\(\s*([^)]*)\)", s)
    if m:
        parts = [p for p in re.split(r"[,\s/]+", m.group(1)) if p]
        try:
            nums = [float(p.rstrip("%")) * (2.55 if p.endswith("%") else 1.0) for p in parts[:3]]
            a = float(parts[3]) if len(parts) > 3 else 1.0
        except Exception:
            return None
        if a > 1.0:
            a = a / 255.0
        return to_hex([int(v + 0.5) for v in nums]), max(0.0, min(1.0, a))
    return None
