# HOVER Adaptation Summary

This document summarizes the changes made to adapt the HOVER implementation to match the HotpotQA structure.

## Files Modified

### 1. run_gepa_hover_compare.py
**Changes:**
- Updated header comment to match HotpotQA style with example command
- Improved argument descriptions to match HotpotQA
- Added comments about "Define the three variants" and "Assign API bases"
- Fixed variant argument handling to check `if bon/itr is not None`
- Added comment "build args for THIS variant"
- Improved live plot loop comments
- Added "return non-zero if any failed" comment
- Changed plot y-axis label to "Best validation recall so far" (HOVER-specific metric)

**Consistency:**
- Same 3 variants as HotpotQA: GEPA, GEPA+merge, GEPA bon=5 itr=5
- Same structure for spawning parallel workers
- Same live plotting and CSV export mechanism

### 2. job.hover_compare.sbatch
**Changes:**
- Updated venv path: `$WORK/venv/hotpotqa2/bin/activate`
- Updated cache paths to match HotpotQA:
  - `$WORK/adBO/cache/hf` (was `$WORK/gepa_cache/hf`)
  - `$WORK/adBO/cache/xdg` (was `$WORK/gepa_cache/xdg`)
- Updated output root: `$WORK/adBO/runs/hover_runs/$RUN_ID`
- Updated working directory: `$WORK/adBO/hover`
- Removed BM25 pre-build section (not needed, handled by worker)
- Added sanity check for vLLM models
- Updated max_metric_calls to 6858 (HOVER-specific budget)

**Consistency:**
- Same SLURM configuration (partition, time, nodes, tasks)
- Same vLLM server setup (3 servers on 3 GPUs)
- Same port allocation strategy
- Same module loading (hpcfund, rocm/6.4.1)

### 3. job.hover_compare_resume.sbatch
**Changes:**
- Updated job name: `hover_gepa_cmp` (from `hotpot_gepa_cmp`)
- Updated output root: `$WORK/adBO/runs/hover_runs/latest` (resume-friendly fixed path)
- Updated working directory: `$WORK/adBO/hover`
- Updated script call: `run_gepa_hover_compare.py`
- Updated max_metric_calls to 6858
- Added `--retriever_threads 8`
- Uses `>>` for vLLM log appending (resume-friendly)
- Supports `CLEAN=1` environment variable to restart fresh

**Resume Features:**
- Uses fixed "latest" folder instead of timestamped directories
- Appends to vLLM logs rather than overwriting
- Timestamps each submission in `last_submit_time.txt`
- GEPA automatically resumes from log_dir checkpoints

### 4. resume_submit.sh
**Changes:**
- Updated job description: "HoVer" (from "HotpotQA")
- Updated sbatch file: `job.hover_compare_resume.sbatch`
- Updated all paths to use `hover_runs/latest`
- Helper script with useful commands for monitoring

**Features:**
- Submits the resume-friendly batch job
- Prints helpful commands for monitoring progress
- Shows how to follow logs, check ports, and clean restart

### 5. run_gepa_hover_worker.py
**No changes needed** - Already well-structured with:
- Default max_metric_calls: 6858 (HOVER-specific)
- Correct metric: hover_recall_score
- Correct data loader: load_hover_splits
- Correct program: HoverMultiHop
- Optional require_num_hops parameter

## Key Differences from HotpotQA (Expected)

1. **Metrics:**
   - HotpotQA: EM (Exact Match) score
   - HOVER: Recall score

2. **Budget:**
   - HotpotQA: 10,000 metric calls
   - HOVER: 6,858 metric calls

3. **Data:**
   - HotpotQA: load_hotpotqa_splits()
   - HOVER: load_hover_splits() with optional require_num_hops

4. **Program:**
   - HotpotQA: HotpotMultiHopQA
   - HOVER: HoverMultiHop

5. **Output directories:**
   - HotpotQA: `$WORK/adBO/runs/hotpotqa_runs/`
   - HOVER: `$WORK/adBO/runs/hover_runs/`

## Testing

### One-time run (with timestamp):
```bash
cd /work1/krishnamurthy/arvind/adBO/hover
sbatch job.hover_compare.sbatch
```

This will:
1. Start 3 vLLM servers on ports 18000+, 18001+, 18002+
2. Run 3 HOVER variants in parallel:
   - GEPA
   - GEPA+merge
   - GEPA bon=5 itr=5
3. Generate live comparison plots and CSV
4. Save all outputs to `$WORK/adBO/runs/hover_runs/<RUN_ID>/logs/`

### Resume-friendly run (fixed "latest" folder):
```bash
cd /work1/krishnamurthy/arvind/adBO/hover
./resume_submit.sh
```

This will:
- Submit a resume-friendly job that uses a fixed output path
- Allow you to resubmit to continue from where you left off
- GEPA will automatically resume from its checkpoints
- Useful for running in stages or recovering from interruptions

To clean and restart:
```bash
cd /work1/krishnamurthy/arvind/adBO/hover
CLEAN=1 sbatch job.hover_compare_resume.sbatch
```

## File Structure

```
adBO/hover/
├── run_gepa_hover_compare.py          # Comparison driver (spawns 3 workers)
├── run_gepa_hover_worker.py           # Single-run worker
├── job.hover_compare.sbatch           # SLURM batch script (timestamped runs)
├── job.hover_compare_resume.sbatch    # SLURM batch script (resume-friendly)
├── resume_submit.sh                   # Helper script for resume submissions
├── hover_data.py                      # Data loading
├── hover_metric.py                    # Metrics and feedback
├── hover_program.py                   # DSPy program
├── wiki_retriever.py                  # BM25 retriever
└── ...
```

## Key Features of Resume Scripts

1. **Fixed Output Path**: Uses `$WORK/adBO/runs/hover_runs/latest` instead of timestamped directories
2. **Log Appending**: vLLM logs use `>>` to append rather than overwrite
3. **Automatic Resume**: GEPA reads its checkpoints from `log_dir` and continues seamlessly
4. **Clean Restart**: Set `CLEAN=1` to remove previous runs while keeping caches
5. **Run Tracking**: Each submission writes to `last_submit_time.txt` for tracking
