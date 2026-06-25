# Copyright 2026 Ritesh Thawkar
# SPDX-License-Identifier: Apache-2.0

"""Compatibility package for BAGEL upstream imports.

This checkout keeps the original BAGEL data modules under ``data_temp`` to
avoid colliding with repo-level experiment data. Upstream BAGEL code imports
them as ``data.*``, so this package preserves that import surface.
"""
