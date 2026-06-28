"""Target-label policy subsystem.

Combines an immutable normalized dataset, its immutable audit artifact, and a tracked
human-authored label-decision configuration into an immutable, content-addressed
target-label policy artifact. This package depends on the dataset integrity contract and
the audit artifact contract; neither of those packages imports label-policy code.
"""
