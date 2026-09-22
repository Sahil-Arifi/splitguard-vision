# Evaluate independently labeled image pairs

`evaluate-labeled-pairs` accepts your own labeled image corpus. It selects a
SHA/pHash decision threshold using a calibration fold, freezes that threshold,
and reports precision, recall, F1, and confusion counts on a separate evaluation
fold. It does not generate labels, fetch a dataset, or tune on evaluation scores.

**No external-corpus results have been published yet.** The committed detector
benchmark still measures generated fixtures. This command provides the missing
evaluation path; its unit tests are synthetic protocol tests, not evidence of
real-world detection quality.

## Build the corpus before looking at detector scores

1. Collect images you have permission to use from the domains you care about.
   Include natural near duplicates, crops, recompression, lighting changes, and
   hard negatives such as similar objects in different scenes.
2. Define identity in writing. For this workflow, a duplicate is an exact copy or
   a transformed version of the same source image. Two different photos of the
   same class are distinct. Resolve ambiguous cases before scoring.
3. Assign every original and all its derivatives one stable source-family ID.
   Split families into calibration and evaluation **before forming pairs**.
   A negative pair uses two different families from the same fold. Never put
   derivatives of a calibration image in evaluation under a different ID.
4. Label each pair independently of detector outputs. Ideally use two reviewers
   and adjudicate disagreements. Keep the reviewer protocol, domain counts,
   exclusions, sampling method, and date with your private dataset.
5. Decide the minimum acceptable calibration precision in advance. Run the
   command once after labels and folds are fixed. Repeatedly redesigning the
   detector against evaluation results turns that fold into calibration data;
   obtain a fresh held-out corpus for the next final evaluation.

There is no minimum sample size enforced beyond requiring both classes in each
fold. Two pairs can test the command but cannot support a deployment claim. Include
enough independently sampled families to measure the failure modes you care about.

## CSV format

Both files require this exact header, UTF-8 text, and no extra columns:

```csv
left,right,label,left_group,right_group
cal/source_01.png,cal/source_01_crop.jpg,1,cal-01,cal-01
cal/source_01.png,cal/source_02.png,0,cal-01,cal-02
```

A separate evaluation file might look like this:

```csv
left,right,label,left_group,right_group
eval/source_03.png,eval/source_03_export.jpg,1,eval-03,eval-03
eval/source_03.png,eval/source_04.png,0,eval-03,eval-04
```

These filenames are format examples, not an included dataset. Supply your own
images and labels. `1` means duplicate, `0` means distinct. Duplicate pairs must
share a source-family ID; distinct pairs must have different IDs. Paths are
relative to one image root and use `/`. Absolute paths, traversal, missing files,
self-pairs, and repeated pairs (including reversed order) are rejected.

Use a dedicated image root, for example `datasets/labeled-pairs/images/`. Keep
the two CSVs beside it. The output must be outside that image root and must not
overwrite either input CSV. Dataset files under `datasets/` and the default
generated artifact are ignored by Git.

## Run

```bash
uv run splitguard evaluate-labeled-pairs \
  datasets/labeled-pairs/calibration.csv \
  datasets/labeled-pairs/evaluation.csv \
  --dataset-root datasets/labeled-pairs/images \
  --label-source "Two reviewers; source-image identity; disagreements adjudicated; protocol v1" \
  --minimum-precision 0.95 \
  --output artifacts/labeled_pair_evaluation.json
```

All scoring is local. This command uses the same `phash64-dct-v1` implementation
as the audit pipeline and does not load DINOv2 weights.

## Threshold selection and evidence

The decision rule is **equal SHA-256 OR pHash distance at most the selected
radius**. The calibration sweep includes SHA-only (`radius = -1`) and radii
0 through 64. Among thresholds meeting the requested empirical calibration
precision, selection maximizes recall, then precision, then prefers the smallest
radius. A SHA-only result can therefore win when perceptual matching adds no
benefit. If none qualifies, the command fails and writes no new result.

The JSON contains the full calibration sweep, the selected radius, exactly one
evaluation result at that radius, both CSV hashes, an image-content snapshot hash,
configuration hash, package versions, and producing Git revision. The CLI prints
the selected threshold and both fold results. The command does not change the
production audit configuration. If you choose to use the threshold there, configure
`phash.hamming_threshold` separately; for SHA-only set `phash.enabled: false` and
keep embedding-only matches as review candidates.

The loader checks that the folds share no declared family, resolved image path,
byte-identical content, or identical full-resolution RGB pixels after the detector's
EXIF and white-background alpha normalization. It also rejects conflicting
family assignments for identical content within a fold. These checks catch common
split mistakes; they cannot find an undeclared crop or other transformed relative.

Calibration precision is not a guarantee of evaluation precision. The evaluation
result may miss that target and is still reported honestly. Pair precision depends
on how positives and negatives were sampled, so it is not a dataset-wide false
positive estimate. This workflow measures pair classification, not approximate
neighbor retrieval, DINOv2 quality, repair optimality, or causal accuracy improvement.

Publish a corpus description, label protocol, counts by domain, calibration rule,
and untouched evaluation results alongside any future headline performance claim.
Do not publish private source imagery without permission.
