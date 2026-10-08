"""Scarcity classes, provenance channels and the judged Scarcity-Provenance matrix (Figure 5)."""

CLASSES = [
    ("O1", "Rarity of condition"),
    ("O2", "Event sparsity"),
    ("O3", "Stratum sparsity"),
    ("E1", "Label poverty"),
    ("E2", "Measurement poverty"),
    ("E3", "Dimensional excess"),
    ("T1", "Temporal obsolescence"),
    ("T2", "Contextual mismatch"),
    ("C1", "Custodial fragmentation"),
    ("C2", "Consent and access limits"),
]

CHANNELS = [
    ("P1", "Pooling"),
    ("P2", "Proxy"),
    ("P3", "Prior"),
    ("P4", "Unlabeled"),
    ("P5", "Procurement"),
    ("P6", "Re-expression"),
]

# Judged potential of each channel against each class: 0 none, 1 weak, 2 moderate, 3 strong.
POTENTIAL = [
    [3, 2, 3, 1, 1, 1],  # O1
    [3, 1, 2, 1, 2, 1],  # O2
    [3, 1, 1, 1, 2, 1],  # O3
    [1, 2, 1, 3, 3, 1],  # E1
    [1, 1, 2, 1, 2, 1],  # E2
    [2, 3, 3, 2, 1, 1],  # E3
    [1, 2, 2, 2, 3, 0],  # T1
    [2, 2, 1, 2, 2, 1],  # T2
    [3, 1, 0, 0, 0, 1],  # C1
    [2, 1, 1, 0, 1, 1],  # C2
]

# Rows after which a family of the scarcity axis ends (O | E | T | C).
FAMILY_BREAKS = [3, 6, 8]
