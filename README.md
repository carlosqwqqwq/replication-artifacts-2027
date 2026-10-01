# Vernier replication package (staging)

This tree is generated from the `rq1-comparison` source tree. It contains the Vernier framework snapshot, six emulator observation schemas, the test-condition catalog, all analysis and experiment support scripts, and an RVT-Bench staging ledger.

## Counting unit

The current Direct crosswalk has **27 unique root-cause behavior points** and **103 trigger events** (36 outcome + 67 state). The event-level rows are therefore not 103 independent defects. `rvt-bench/defects.jsonl` keeps the root-cause rows; `rvt-bench/events.jsonl` keeps the 103 release rows.

## Missing publication metadata

`event_id` joins and upstream issue/PR URLs are explicitly marked pending where the current crosswalk does not provide them. Do not fill them with guessed URLs. The public source repository is listed in `ANONYMIZED_URL.txt`; the anonymous mirror still needs the GitHub OAuth step.

## Large data

Per-test coverage profiles and instruction traces remain in the external run roots because a single raw-only run is tens of GiB. Use `scripts/index_per_test_artifacts.py` to emit one JSONL record per case/target/job, including profile paths and hashes plus executed-PC traces; the package must publish that index together with the referenced profile files.
