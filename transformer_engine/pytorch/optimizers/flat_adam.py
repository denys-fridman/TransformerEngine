# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Segment table of the flat Adam kernel."""
from typing import Iterable, Tuple

import torch


def flat_adam_segments(
    pieces: Iterable[Tuple[int, int, int]], numel: int, device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Segment table of ``flat_adam_capturable_master`` for one contiguous buffer.

    ``pieces`` are ``(offset, numel, param_group)`` runs of the buffer, disjoint; elements no
    run covers are padding and left untouched by the kernel. Returns ``(seg_starts, seg_groups)``:
    the sorted int64 element offsets of the segments, with the buffer size appended, and the
    int32 param group of each segment, ``-1`` for padding. Adjacent runs of one param group
    form a single segment.
    """
    starts, groups = [], []
    end = 0
    for offset, count, group in sorted(pieces):
        if count == 0:
            continue
        if offset < end:
            raise ValueError(f"Overlapping pieces at element {offset}")
        if group < 0:
            raise ValueError(f"Negative param group {group}")
        if offset > end:
            starts.append(end)
            groups.append(-1)
        if not groups or groups[-1] != group:
            starts.append(offset)
            groups.append(group)
        end = offset + count
    if end > numel:
        raise ValueError(f"Pieces extend to element {end} of a {numel}-element buffer")
    if end < numel or not groups:
        starts.append(end)
        groups.append(-1)
    starts.append(numel)
    return (
        torch.tensor(starts, dtype=torch.int64, device=device),
        torch.tensor(groups, dtype=torch.int32, device=device),
    )
