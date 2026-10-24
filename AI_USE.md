# AI use log

The project brief requires full disclosure of material AI use. This file is
a factual record of what the AI produced. It is **not** the AI reflection
required in the report: that must be written by the student, in their own
words, covering why AI was used, how its output was checked, which
decisions were the student's, and how it affected their learning.

## Tool

Claude (Anthropic), model Claude Opus 5.5, used through the claude.ai app
in October 2026.

## What the AI did

| Date | Request by student | AI contribution |
|---|---|---|
| 2026-10-25 | Is "reconstructing missing temperature" a good project? Give a guide. | Read the brief and dataset; audited the data (found only 2 of 5 sensors usable, ~76% duplicate rows, misleading file name); proposed scope, method ladder, validation design and repo layout. |
| 2026-10-25 | Shorten and improve the student's Stage 1 script (879 lines). | Refactored into documented functions (301 lines); fixed an x-axis tick bug; converted to local time; identified that the 1.5x gap threshold discarded ~99% of data. |
| 2026-10-25 | Write all the code. | Wrote the `tempfill` package, `main.py`, the test suite, README and this log. Ran all experiments on the provided data. |

## Approximate level of AI contribution

* Code (`tempfill/`, `main.py`, `tests/`): almost entirely AI-generated.
* Experimental design (three-way split, synthetic masking with partial
  observation, acceptance criterion, method ladder): AI-proposed.
* Original Stage 1 exploration script: written by the student before AI
  refactoring.

## Issues found during development (useful for the reflection)

These were found by running the code on the real data, not by inspection:

1. The first validation design required fully observed gap windows. On
   the real data no stretch longer than about 5 hours is fully observed,
   so long synthetic gaps could not be placed. Redesigned to allow 80%
   observed windows and score only known slots.
2. The first boosted-tree model was poor on short gaps (MAE 0.19 vs 0.07
   for linear) because training rows scale with gap length. Fixed by
   weighting every gap equally.
3. The first hybrid-switch rule averaged MAE across gap lengths, which hid
   a real loss at 3-hour gaps. Replaced by a per-length tolerance check.
4. The first PCHIP-vs-spline test assumed a spline overshoots any step; it
   only does so near a ramp-to-plateau corner. Test corrected.
5. pandas 3 returns read-only arrays, which broke in-place filling. Fixed
   with explicit copies.
6. The acceptance threshold of 0.5 °C is an **assumption**, not taken from
   the sensor datasheet. It should be checked against the NEMo sensor
   specification.

## What the student still needs to verify and own

* Run `python main.py` and `pytest` themselves and confirm the numbers.
* Read every module and be able to explain each method, the validation
  design and the performance results without AI (required for the
  interview).
* Check the acceptance threshold against the sensor datasheet.
* Write the report and the AI reflection.
