"""datapipe — dataset-aware processing & moving pipeline for lab acquisition data.

Home of the serial-section dataset model/loader and the processing stages
(registration now; segmentation / compress-handoff / namespace-finalize to come).
"""

from __future__ import annotations

from datapipe.models.serialsection import (
    SerialSectionSample,
    find_samples,
    load_sample,
)
from datapipe.register import (
    RegistrationPlan,
    RegistrationResult,
    plan_registration,
    register_sample,
    run_registration,
)
from datapipe.compress import compress_commands, resolve_datapack
from datapipe.segment import launch_segment, registration_dir, resolve_registration

__all__ = [
    "SerialSectionSample",
    "find_samples",
    "load_sample",
    "RegistrationPlan",
    "RegistrationResult",
    "plan_registration",
    "register_sample",
    "run_registration",
    "launch_segment",
    "registration_dir",
    "resolve_registration",
    "compress_commands",
    "resolve_datapack",
]
