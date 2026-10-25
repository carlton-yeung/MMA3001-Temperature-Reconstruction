# AI use log

The project brief requires full disclosure of material AI use. This file is
a factual record of how AI was used. It is **not** the AI reflection
required in the report: that must be written by the student, in their own
words, covering why AI was used, how its output was checked, which
decisions were the student's, and how it affected their learning.

## Tool

Claude (Anthropic), model Claude Opus 5.5, used through the claude.ai app
in October 2026.

## How the project was developed

| Phase | Student | AI |
|---|---|---|
| Problem and dataset | Chose Dataset 2 and the topic (reconstructing missing temperature readings). | Read the brief, audited the data (2 of 5 sensors usable, 76% duplicate rows, misleading file name, clock drift). |
| Initial exploration | Wrote the original Stage 1 script (`archive/stage1_exploration_original.py`). | Refactored it and identified that its 1.5x gap threshold discarded most of the data. |
| First complete version | Asked the AI to write the code. | Wrote a full first version (package, tests, docs) with AI-chosen design decisions, and ran it on the data. |
| **Design review (decisions 1-14 in the README)** | **Made every design decision with written reasoning**, stage by stage: primary sensor, scope, time grid, validation design, metrics, acceptance criterion, baselines, methods, selection protocol, performance focus, limitations, future work. | Asked one question per stage with options and facts from the data, then challenged or refined the student's reasoning before the next stage. Suggested refinements the student accepted or declined (e.g. the student initially chose an interpolation-only scope, then switched to include harmonic least squares after discussion). |
| Rebuild | Approved the final decision log. | Rewrote the code, tests, figures and README to implement the student's decisions, and ran all experiments. |

## Approximate level of AI contribution

* **Code** (`tempfill/`, `main.py`, `tests/`): almost entirely AI-generated.
* **Design decisions** (which data, scope, validation design, metrics,
  acceptance criterion, baselines, methods, selection rules, performance
  focus): made by the student in the design review, with AI-supplied
  options and critique. The decision log in the README records them.
* **Data facts and the NEMo specification**: found by the AI; the
  specification came from a reseller's product page and still needs
  confirming for Monash's units.

## Issues found in AI output during development

Found by running the code on the real data, not by inspection:

1. The first validation design required fully observed gap windows; none
   longer than about 5 hours exist in this data. Redesigned (and later
   formalised by the student as two designs with an agreement check).
2. A boosted-tree model in the first version was poor on short gaps
   because training rows scaled with gap length. (Removed from scope in
   the design review.)
3. The first hybrid-switch rule averaged MAE over gap lengths, hiding a
   real loss at 3-hour gaps. Replaced by a per-length rule.
4. A test assumed a spline overshoots any step; it only does so near a
   ramp-to-plateau corner. Test corrected.
5. pandas 3 read-only arrays and pandas 2.2 index alignment both broke
   code paths; fixed and verified in a clean environment.
6. The flop estimate charged the hybrid for building its design matrix
   even when it fitted no gaps, giving a meaningless 6.5 GFLOP/s figure.
   Fixed.
7. README claims in the first version were checked against the result
   tables; several were overstated (e.g. "higher K overfits" when the
   results were flat) and were corrected.

## What the student still needs to verify and own

* Run `python main.py` and `pytest` and confirm the numbers.
* Read every module and be able to explain each method, the validation
  design and the performance results without AI (required for the
  interview).
* Confirm the NEMo accuracy specification for Monash's sensors.
* Write the report and the AI reflection.
