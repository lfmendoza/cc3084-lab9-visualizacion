# ADR 0001. Adopt non-resident international visitors as the reference population

- **Status:** accepted
- **Date:** July 2026
- **Scope:** definition of the indicator, construction of the series, historical
  comparability

## Context

The migration database changed its recording criterion in 2023 in two
simultaneous respects, one documented and one not.

The first, stated in the notes of the file itself, excluded travellers without a
tourist purpose from the count. The second is documented nowhere: from January
2023 the register drops arrivals of people whose country of residence is
Guatemala, who until 2022 were counted as tourists and excursionists despite not
being international visitors under the definition of the World Tourism
Organization.

The magnitude of the second change is not marginal. In the years before the
pandemic, residents accounted for around four out of every ten arrivals counted
as tourists and excursionists. A series that includes them until 2022 and
excludes them from 2023 shows an abrupt fall at the boundary between the two
regimes that corresponds to no tourism phenomenon: it is a measurement artefact.

## Alternatives considered

1. **Use the total of tourists and excursionists as published.** This is the
   literal reading of the file and the one the previous advance produced. It has
   the advantage of requiring no decision and the fatal defect of comparing
   different populations with each other.
2. **Restrict the analysis to the period after 2023.** This removes the
   comparability problem, but leaves fourteen months of history and makes it
   impossible to estimate any seasonal component or to measure the effect of the
   pandemic.
3. **Exclude residents across the whole period.** This requires rebuilding the
   historical series under the current criterion, which the detail by country of
   residence allows without ambiguity.

## Decision

The third alternative is adopted. The reference population of the study is
non-resident international visitors: tourists and excursionists whose declared
country of residence is other than Guatemala, throughout the period from January
2009 to June 2026.

The only exception is the sea entry series, built on the total reported for the
reasons set out in ADR 0002.

## Consequences

- The seven series share a single definition of population, which makes them
  comparable with each other and over time.
- The figures of this study do not match those published officially for the
  years before 2023, and the difference is large. The report warns of this
  explicitly in its first section and quantifies the adjustment year by year.
- The decision is embedded in the processed dataset as the
  `reference_population` column, so that any consumer can reproduce both the
  comparable series and the published one.
- The pipeline keeps the excluded categories in the fact detail: nothing is
  discarded, it is only flagged. Reverting the decision does not require
  reprocessing the original file.
