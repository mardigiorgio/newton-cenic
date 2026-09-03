# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The capture-safe chunked scan against a numpy reference, with and without
the device-side count bound the collision pipeline relies on."""

import unittest

import numpy as np
import warp as wp

from newton._src.geometry.capture_safe_scan import SCAN_CHUNK, capture_safe_scan_int32

SENTINEL = -7


class TestCaptureSafeScan(unittest.TestCase):
    def setUp(self):
        self.device = wp.get_device("cuda:0") if wp.is_cuda_available() else wp.get_device("cpu")
        self.rng = np.random.default_rng(1234)

    def _check(self, vals_np, inclusive, count, extra, in_place):
        n = len(vals_np)
        ref = np.cumsum(vals_np, dtype=np.int64)
        if not inclusive:
            ref = ref - vals_np
        ref = ref.astype(np.int32)
        vals = wp.array(vals_np, dtype=wp.int32, device=self.device)
        out = vals if in_place else wp.full(n, SENTINEL, dtype=wp.int32, device=self.device)
        count_arr = None if count is None else wp.array([count], dtype=wp.int32, device=self.device)
        capture_safe_scan_int32(vals, out, inclusive=inclusive, count=count_arr, count_extra=extra)
        got = out.numpy()
        n_eff = n if count is None else min(n, max(count, 0) + extra)
        np.testing.assert_array_equal(got[:n_eff], ref[:n_eff])
        # Past the bound a slot is either untouched or carries the exact full-scan value.
        tail = got[n_eff:]
        untouched = tail == (vals_np[n_eff:] if in_place else SENTINEL)
        exact = tail == ref[n_eff:]
        self.assertTrue(np.all(untouched | exact))
        if count is not None and n_eff + SCAN_CHUNK < n:
            # Whole chunks past the bound are never written.
            first_skipped = ((n_eff + SCAN_CHUNK - 1) // SCAN_CHUNK) * SCAN_CHUNK
            self.assertTrue(np.all(untouched[first_skipped - n_eff :]))

    def test_matches_numpy(self):
        for n in (1, 5, 255, 256, 257, 1000, 4 * SCAN_CHUNK + 3, 70000):
            vals_np = self.rng.integers(0, 100, size=n, dtype=np.int32)
            for inclusive in (True, False):
                for count in (None, 0, 1, n // 3, n - 1, n, n + 5):
                    for extra in (0, 1):
                        for in_place in (False,) if inclusive else (False, True):
                            with self.subTest(n=n, inclusive=inclusive, count=count, extra=extra, in_place=in_place):
                                self._check(vals_np, inclusive, count, extra, in_place)

    def test_negative_count_is_empty(self):
        vals_np = self.rng.integers(0, 100, size=1000, dtype=np.int32)
        self._check(vals_np, True, -3, 1, False)


if __name__ == "__main__":
    unittest.main()
