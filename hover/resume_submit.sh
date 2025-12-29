#!/bin/bash
set -euo pipefail

echo "Submitting resume-friendly HoVer compare job..."
jid=$(sbatch --parsable job.hover_compare_resume.sbatch)
echo "Submitted: $jid"

echo
echo "Follow logs (after redirect, check OUT_ROOT):"
echo "  tail -f $WORK/adBO/runs/hover_runs/latest/slurm_${jid}.out"
echo
echo "Live outputs folder:"
echo "  $WORK/adBO/runs/hover_runs/latest/logs"
echo
echo "Ports used (once job starts):"
echo "  cat $WORK/adBO/runs/hover_runs/latest/ports.txt"
echo
echo "vLLM server logs:"
echo "  ls $WORK/adBO/runs/hover_runs/latest/vllm_*.log"
echo
echo "To clean and restart fresh:"
echo "  CLEAN=1 sbatch job.hover_compare_resume.sbatch"
