#!/usr/bin/env python3
"""Let the upstream Inkling 2 unit tests run on sm_8x.

The tests under tests/models/inkling2 skip themselves off Hopper / Blackwell.
This widens those gates (anchored edits, fails on a mismatch) and scales the
tiny-model memory fractions, which were written for a 180 GiB B200, to a
24 GiB card.

Usage: widen_test_gates.py <dir containing tests/models/inkling2>
"""
import pathlib
import sys

GATE_OLD = "    or current_platform.is_device_capability_family(120)\n):\n    pytest.skip("
GATE_NEW = ("    or current_platform.is_device_capability_family(120)\n"
            "    or current_platform.is_device_capability_family(80)  # sm86-port\n):\n    pytest.skip(")
EDITS = {
    "test_moe.py": [
        (GATE_OLD, GATE_NEW),
        ('    not current_platform.is_device_capability(90), reason="Humming experts (Hopper)"\n',
         "    not (current_platform.is_device_capability(90) or current_platform.is_device_capability_family(80)),\n"
         '    reason="Humming experts (Hopper)",  # sm86-port\n'),
    ],
    "test_model.py": [
        (GATE_OLD, GATE_NEW),
        ("gpu_memory_utilization=0.10 if looped else 0.035", "gpu_memory_utilization=0.30 if looped else 0.15"),
        ("gpu_memory_utilization=0.035,", "gpu_memory_utilization=0.15,"),
    ],
    "test_dspark.py": [(GATE_OLD, GATE_NEW)],
    "test_attention.py": [
        ("    _cap is None or _cap.major not in (9, 10, 11, 12),\n",
         "    _cap is None or _cap.major not in (8, 9, 10, 11, 12),  # sm86-port\n"),
    ],
}

root = pathlib.Path(sys.argv[1]) / "tests" / "models" / "inkling2"
for name, edits in EDITS.items():
    path = root / name
    text = path.read_text()
    for old, new in edits:
        if old not in text:
            sys.exit(f"ANCHOR MISMATCH in {name}: {old!r}")
        text = text.replace(old, new)
    path.write_text(text)
    print("widened", name)
