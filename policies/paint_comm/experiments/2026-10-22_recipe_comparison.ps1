# Overnight comparison (2026-10-22): all arms continue from memory_st on the
# corrected simulator, choose checkpoints on maps/validation.
# The first pass (night_*) was spoiled by a GPU memory leak in checkpoint
# selection (fixed in torch_sim.capture_graph); the rerun is night2_*.
$ErrorActionPreference = "Continue"
Set-Location (Resolve-Path "$PSScriptRoot\..\..\..")   # repository root
$prefix = "night2"
$common = "--minutes 100 --init-from trained/paint_comm/checkpoints/memory_st_latest.pt --custom-maps maps/training " +
          "--select-maps maps/validation --custom-weight 0.7 --real-weight 0 --random-maps 128 " +
          "--colour-incentives --colour-diversity-coef 0.01 --incentive-floor 0.2 --anneal-updates 1 " +
          "--game-seconds 120 --game-seconds-final 120 --trainable-memory --memory-hardness-start 1 " +
          "--eval-episodes 4"
$arms = [ordered]@{
    "base"    = ""
    "team"    = "--team-share 0.5"
    "bptt64"  = "--bptt 64 --minibatches 4"
    "explore" = "--colour-entropy-target 0.2 --memory-noise 0.02"
    "all"     = "--team-share 0.5 --bptt 64 --minibatches 4 --colour-entropy-target 0.2 --memory-noise 0.02"
}
foreach ($arm in $arms.Keys) {
    $run = "${prefix}_$arm"
    "$(Get-Date -Format s) start $run" | Out-File -Append -Encoding utf8 out\paint\overnight_progress.txt
    Invoke-Expression "python -m policies.paint_comm.train --run $run $common $($arms[$arm]) *> out\paint\${run}_console.txt"
}
foreach ($arm in $arms.Keys) {
    $run = "${prefix}_$arm"
    "$(Get-Date -Format s) report $run" | Out-File -Append -Encoding utf8 out\paint\overnight_progress.txt
    python -m tools.paint_report --policy paint_comm --checkpoint out/paint/$run/best.pt --maps maps/validation --episodes 8 *> out\paint\${run}_validation_report.txt
}
"$(Get-Date -Format s) done" | Out-File -Append -Encoding utf8 out\paint\overnight_progress.txt
