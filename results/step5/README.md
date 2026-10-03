# Step 5 result files

- `resident_summary.csv`: one concise row per canonical symmetric Nash resident.
- `resident_mutant_pair_results.csv.gz`: complete stage-, pair-, and focal-position-resolved results.
- `complete_results.json.gz`: nested machine-readable search objects, including composition payoffs and optimizer candidates.
- `symmetric_nash_candidates.csv`: Stage A resident screening output.
- `continuous_optimizer_runs.csv`: seeds, objectives, convergence messages, and optimizer diagnostics.
- `landscape_points.csv`: values used to draw the four invasion landscapes.
- `figures/`: one important-resident mutant landscape for each `K=2,3,4,5`.
- `run_metadata.json`: authoritative baseline, command, environment, parameters, and seed.

The classification is determined from the rare-limit polynomial. The plotted
and summary maximum at `epsilon=0.01` is a declared diagnostic, not the ESS
criterion.
