# Examples

The repository ships no real dataset. Run the deterministic demo to create a small,
privacy-safe ImageFolder fixture locally:

```bash
uv sync --frozen
uv run splitguard demo
```

The ignored `demo-data/dataset/` tree contains an exact train/test copy, a JPEG
recompression, a resize, a conflicting-label copy, and a malformed file. The command
audits it, performs group-aware repair, verifies that every source byte is unchanged,
and writes its local review bundle to ignored `artifacts/demo/`.

To regenerate the committed report from the retained raw results:

```bash
uv run splitguard report \
  --audit artifacts/audit.json \
  --repair artifacts/repair.json \
  --detection-benchmark artifacts/detection_benchmark.json \
  --scaling-benchmark artifacts/scaling_benchmark.json \
  --training-results artifacts/training_results.json \
  --dataset-root demo-data/dataset
```

For sensitive datasets, omit `--dataset-root` or pass `--no-thumbnails`. Relative
paths and stable IDs remain in the report; source images are not uploaded or modified.
