# AI Use Log

This log records material AI use during the project, as required by the project brief. It is the evidence base for the *AI use and reflection* section of the report.

**Tool:** Claude (Anthropic), used through claude.ai, in one continuous conversation from 11 to 24 October 2026.



## Log

| Date | What I asked for | What the AI produced | How I checked it / what I changed |  Decisions that were mine |
|---|---|---|---|---|
| 11 Oct | Explain project brief and initial structure | Summary of deliverables, marking criteria, and suggested step-by-step plan | Cross-referenced against unit guide requirements | Decided the overall execution timeline |
| 11 Oct | Dataset evaluation and comparison logic | Comparison of 3 datasets (recommended Dataset 2) | Reviewed class distributions and feature fits | Overrode recommendation and selected Dataset 3 for better domain relevance |
| 11 Oct | Code for dataset download and data exploration | Notebook 01: class counts, box sizes, image viewer, perceptual hash leak checks | Executed code, handled local pathing, and verified hash outputs | Filtered out invalid bounding boxes before proceeding |
| 12 Oct | Solution for frame leakage in dataset splits | Notebook 02: blocked split logic and baseline YOLOv11n training script | Audited generated split boundaries to ensure zero temporal leakage | Adopted the blocked split strategy |
| 24 Oct | Code for alternative model comparison & metric sweeps | Notebook 03: YOLOv11s setup, threshold sweeps, bootstrap intervals, GPU/CPU profiling | Ran code locally; modified threshold sweep step size and fixed timing logic | Selected YOLOv11n for final deployment; accepted 5% miss-rate target |
| 24 Oct | Repository structure and packaging code | `src/porkqc` package, `pytest` unit tests, README, docs, and log template | Checked test coverage, added missing edge-case tests, and refactored imports | Finalized modular structure and repository layout |

## Errors, limitations and unhelpful suggestions from the AI

Recorded as they happened:

1. **Wrong guess about a class label.** Claude first suggested 'packaging-error' might be a catch all label on top of other defects. The co-occurrence analysis showed it appears alone in 1,498 of 1,580 images, so this was wrong.

2. **Wrong prediction about leakage.** The AI expected the leaky original split to give a higher score than the blocked split. The measured scores were similar (0.43 vs 0.46). 

3. **Code that needed manual editing.** The first version of notebook 01 needed lines to be uncommented by hand, which caused an `IndentationError`. It was later changed to run without edits.

4. **Code warning.** The first version of notebook 02's split code produced a NumPy warning about shuffling a pandas array. This was fixed before the notebook was used.

5. **Scope beyond the taught content.** The object-detection model (YOLO) was not covered in the unit. Its internals were not derived or checked in this project; it was used as an external, documented library.



## How AI-generated code was verified

- Each notebook was tested by the AI on a synthetic dataset before use, then run by me on the real data in Google Colab.
- 
- Core functions were moved into `src/porkqc` and covered by 58 pytest tests (known answers, convergence, comparison with the Ultralytics reference implementation, and regression against saved results).
- The package's split function was checked to reproduce notebook 02's split exactly.

- Looked at the images with their boxes. When I filtered for packaging-error, unsealed boxes showed up too, so I asked why. That led to a per-split overlap check: in train, 57 of the 1,541 packaging-error images also had unsealed; in valid it was 3 of 23, and in test 3 of 16.
