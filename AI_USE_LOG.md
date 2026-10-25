# AI Use Log
 
This log records material AI use during the project, as required by the project brief. It is the evidence base for the *AI use and reflection* section of the report.
 
**Tool:** Claude (Anthropic), used through claude.ai, in one continuous conversation from 11 to 25 October 2026.
 
## Log
 
| Date | What I was working on | What the AI produced | How I checked it / what I changed | Decisions that were mine |
|---|---|---|---|---|
| 11 Oct | Understanding the project brief and planning the work | Summary of deliverables, marking criteria and a suggested plan | Checked it against the brief and the unit schedule | – |
| 11 Oct | Choosing between the three provided datasets | Comparison of the datasets, recommending Dataset 2 | – | Chose Dataset 3 instead |
| 11 Oct | Exploring the data: class balance, label co-occurrence, box sizes and possible leakage | Notebook 01: class counts, co-occurrence, box sizes, image viewer, leakage checks | Downloaded the data from Roboflow and ran the notebook in Colab; fixed an IndentationError in the setup cell. Filtered the image viewer by class and noticed unsealed boxes in packaging-error images, which led to the overlap check | – |
| 12 Oct | Fixing the leakage and training a first model | Notebook 02: blocked split, YOLO11n trained on both splits | Ran it on a Colab GPU; reviewed the confusion matrices and score tables | Adopted the blocked split |
| 24 Oct | Comparing an alternative model, setting a pass/fail threshold and measuring speed | Notebook 03: YOLO11s, threshold sweep and selection, bootstrap intervals, GPU/CPU timing | Ran it on a Colab GPU; reviewed the result files and figures | Accepted the 5 % miss-rate target; recommended YOLO11n |
| 24 Oct | Packaging the code as a tested, documented repository | `src/porkqc` package, pytest tests, README, docs, licence and log template | Uploaded it in stages; checked the README numbers against the re-run results | – |
| 25 Oct | Making the notebooks easier to read, and wording some log entries | Same notebook code with plain headings and comments | Re-ran all three notebooks in Colab; accuracy results were identical to before and timings within a few percent | Asked for the simpler style; edited the notebook 3 setup cell (zip check, clearer error messages) |
 
## AI errors and how they were caught
 
Recorded as they happened:
 
1. **Incorrect assumption about a class label.** Claude first suggested `packaging-error` might be a catch-all label on top of other defects. The co-occurrence analysis showed it appears alone in 1,498 of 1,580 images, so the assumption was wrong and was not used.
2. **Incorrect prediction about leakage.** The AI expected the leaky original split to give a higher score than the blocked split. The measured scores were similar (0.43 vs 0.46).
3. **Code that needed manual editing.** The first version of notebook 01 needed lines to be uncommented by hand, which caused an `IndentationError`. It was later changed to run without edits.
4. **Code warning.** The first version of notebook 02's split code produced a NumPy warning about shuffling a pandas array. This was fixed before the notebook was used.
## How AI-generated code was verified
 
- Each notebook was tested by the AI on a synthetic dataset before use, then run by me on the real data in Google Colab.
- Core functions were moved into `src/porkqc` and covered by 58 pytest tests (known answers, convergence, comparison with the Ultralytics reference implementation, and regression against saved results).
- The package's split function was checked to reproduce notebook 02's split exactly.
- I looked at the images with their boxes. When I filtered for `packaging-error`, `unsealed` boxes showed up too, so I asked why. That led to a per-split overlap check: in train, 57 of the 1,541 `packaging-error` images also had `unsealed`; in valid it was 3 of 23, and in test 3 of 16.
- When the class filter in the image viewer didn't take effect, I found the correct setting (`CLASS_TO_SHOW`), reverted my other edit and re-ran the cell.
- After the notebooks were rewritten in a simpler style, I re-ran all three and compared the results with the earlier run: the accuracy numbers matched exactly and only the timings changed slightly.
 
