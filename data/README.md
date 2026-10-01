# Per-test raw artifacts

Run `../scripts/index_per_test_artifacts.py` against a published raw-only run.
The index records one row per target/case/job and points to the original
coverage raw files and instruction traces. Payloads are deliberately not
aggregated into one summary file.
