# ADR 0002. Build the sea entry series on the total reported

- **Status:** accepted
- **Date:** July 2026
- **Scope:** definition of the sea entry series

## Context

ADR 0001 sets non-resident international visitors as the reference population,
that is, non-resident tourists and excursionists. Applied without nuance, that
definition degrades the sea entry series until it becomes useless: since 2017
cruise traffic is recorded under the `Cruceristas` category, outside the
reference population, and in several months the resulting series is exactly
zero.

A series with structural zeros admits no logarithmic transformation, produces
undefined percentage errors and renders any seasonal decomposition meaningless.
The problem is not statistical but definitional: the series no longer measures
what its name states.

## Alternatives considered

1. **Keep the common definition and accept the zeros.** This preserves formal
   homogeneity at the price of publishing a series that does not represent the
   phenomenon and whose forecasts admit no interpretation.
2. **Exclude the sea entry mode from the study.** The laboratory requires
   analysing the three entry modes, so the exclusion would fail the requested
   scope, besides hiding information that matters to management.
3. **Build the series on the total reported for that entry mode.** This recovers
   the continuity of the phenomenon at the price of breaking the homogeneity of
   population across series.

## Decision

The sea entry series is built on the total reported for that entry mode,
including every traveller category. It is the only series of the study with a
different population.

The exception is declared in three places so that no reader can miss it: in the
series profile, in the footnote of every sea entry figure and in the scope
section of the executive report.

## Consequences

- The sea entry series recovers continuity and admits the same methodological
  treatment as the others.
- Its levels are not comparable with those of the other six series. Comparisons
  across series are always made on relative indicators — seasonal strength,
  volatility, percentage fall — and never on absolute volumes.
- The `population` attribute of each series definition records the difference
  programmatically, and the automated tests verify that the exception remains
  the only one.
