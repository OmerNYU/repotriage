"""Heuristic suitability policy, kept strictly separate from objective analysis.

These thresholds are versioned heuristics tied to ``AUDIT_VERSION``, not universal
scientific rules. They are expected to be tuned; tuning would bump the audit version
(and, if thresholds become configurable, audit identity would need to include a
configuration identity). No single opaque aggregate "quality score" is produced.

Boundary semantics are explicit: "insufficient"/"limited" warnings use strict ``<``,
and "high"/"severe"/"low completeness" warnings use strict ``>``. A measured value
exactly equal to its threshold never fires.
"""

from __future__ import annotations

from repotriage.audit.analyzer import AuditAnalysis
from repotriage.audit.models import SuitabilityWarning

MIN_LABELLED_ISSUES = 500
MAX_UNLABELLED_FRACTION = 0.20
MIN_CALENDAR_SPAN_MONTHS = 6
MAX_RARE_LABEL_FRACTION = 0.50
MAX_SHORT_BODY_FRACTION = 0.30


def build_warnings(analysis: AuditAnalysis) -> list[SuitabilityWarning]:
    """Derive suitability warnings from objective metrics, sorted by code ascending."""
    summary = analysis.repository_summary
    labels = analysis.label_metrics
    temporal = analysis.temporal_metrics
    structural = analysis.text_metrics.structural

    warnings: list[SuitabilityWarning] = []

    if summary.labelled_issues < MIN_LABELLED_ISSUES:
        warnings.append(
            SuitabilityWarning(
                code="INSUFFICIENT_LABELLED_ISSUES",
                severity="high",
                value=float(summary.labelled_issues),
                threshold=float(MIN_LABELLED_ISSUES),
                message=(
                    f"Only {summary.labelled_issues} labelled issues; supervised triage "
                    f"modeling typically needs at least {MIN_LABELLED_ISSUES} to train and "
                    "evaluate per-label classifiers reliably."
                ),
            )
        )

    if summary.unlabelled_fraction > MAX_UNLABELLED_FRACTION:
        warnings.append(
            SuitabilityWarning(
                code="HIGH_UNLABELLED_RATE",
                severity="medium",
                value=summary.unlabelled_fraction,
                threshold=MAX_UNLABELLED_FRACTION,
                message=(
                    f"{summary.unlabelled_fraction:.1%} of issues are unlabelled, exceeding "
                    f"the {MAX_UNLABELLED_FRACTION:.0%} guideline; label coverage may bias signal."
                ),
            )
        )

    if temporal.calendar_span_months < MIN_CALENDAR_SPAN_MONTHS:
        warnings.append(
            SuitabilityWarning(
                code="LIMITED_TEMPORAL_COVERAGE",
                severity="medium",
                value=float(temporal.calendar_span_months),
                threshold=float(MIN_CALENDAR_SPAN_MONTHS),
                message=(
                    f"Issues span only {temporal.calendar_span_months} calendar month(s) "
                    f"end to end; at least {MIN_CALENDAR_SPAN_MONTHS} are recommended to "
                    "reflect drift."
                ),
            )
        )

    if labels.unique_label_count > 0:
        rare_fraction = labels.rare_label_buckets.lt_10 / labels.unique_label_count
        if rare_fraction > MAX_RARE_LABEL_FRACTION:
            warnings.append(
                SuitabilityWarning(
                    code="SEVERE_LABEL_LONG_TAIL",
                    severity="medium",
                    value=rare_fraction,
                    threshold=MAX_RARE_LABEL_FRACTION,
                    message=(
                        f"{rare_fraction:.1%} of labels appear in fewer than 10 issues; "
                        "such rare classes are hard to learn."
                    ),
                )
            )

    if structural.short_bodies_lt_100.fraction > MAX_SHORT_BODY_FRACTION:
        warnings.append(
            SuitabilityWarning(
                code="LOW_TEXT_COMPLETENESS",
                severity="medium",
                value=structural.short_bodies_lt_100.fraction,
                threshold=MAX_SHORT_BODY_FRACTION,
                message=(
                    f"{structural.short_bodies_lt_100.fraction:.1%} of issue bodies are shorter "
                    f"than 100 characters, exceeding the {MAX_SHORT_BODY_FRACTION:.0%} guideline."
                ),
            )
        )

    warnings.sort(key=lambda warning: warning.code)
    return warnings
