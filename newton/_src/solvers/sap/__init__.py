# SPDX-License-Identifier: Apache-2.0
"""SAP convex-contact solver, vendored from ``sap_warp`` via a sys.path shim.

``SolverSAP`` is the SAP (precursor to ICF) convex compliant-contact solver; it
lives in an external ``sap_warp`` checkout that has no installable package, so we
add its root to ``sys.path`` (overridable with ``SAP_WARP_PATH``) and re-export
the solver + Newton converters here. ``SolverSAPAdaptive`` wraps ``SolverSAP``
under the shared step-doubling controller (even + global tiling only).
"""

import os
import pathlib
import sys

# Layout contract: sap_warp is a sibling checkout of this repository;
# SAP_WARP_PATH overrides for other layouts.
_sap_root = os.environ.get("SAP_WARP_PATH", str(pathlib.Path(__file__).resolve().parents[4].parent / "sap_warp"))
if _sap_root not in sys.path:
    sys.path.insert(0, _sap_root)

from sim.sap_runtime import (  # noqa: E402
    sap_contacts_from_newton,
    sap_control_from_newton,
    sap_model_from_newton,
    sap_state_from_newton,
)
from sim.solver_sap import SolverSAP  # noqa: E402

from .solver_sap_adaptive import SolverSAPAdaptive  # noqa: E402

__all__ = [
    "SolverSAP",
    "SolverSAPAdaptive",
    "sap_contacts_from_newton",
    "sap_control_from_newton",
    "sap_model_from_newton",
    "sap_state_from_newton",
]
