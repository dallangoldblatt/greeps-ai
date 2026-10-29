# Long run of the "all" recipe (2026-10-23), continuing night2_all for 6 hours.
# Selection on maps/validation.
$ErrorActionPreference = "Continue"
Set-Location (Resolve-Path "$PSScriptRoot\..\..\..")   # repository root
$run = "long_all"
"$(Get-Date -Format s) start $run" | Out-File -Append -Encoding utf8 out\paint\overnight_progress.txt
python -m policies.paint_comm.train --run $run --minutes 360 --init-from trained/paint_comm/checkpoints/night2_all_latest.pt `
    --custom-maps maps/training --select-maps maps/validation --custom-weight 0.7 --real-weight 0 `
    --random-maps 128 --colour-incentives --colour-diversity-coef 0.01 --incentive-floor 0.2 `
    --anneal-updates 1 --game-seconds 120 --game-seconds-final 120 --trainable-memory `
    --memory-hardness-start 1 --eval-episodes 4 `
    --team-share 0.5 --bptt 64 --minibatches 4 --colour-entropy-target 0.2 --memory-noise 0.02 `
    *> out\paint\${run}_console.txt
"$(Get-Date -Format s) report $run" | Out-File -Append -Encoding utf8 out\paint\overnight_progress.txt
foreach ($ckpt in "best", "latest") {
    python -m tools.paint_report --policy paint_comm --checkpoint out/paint/$run/$ckpt.pt `
        --maps maps/validation --episodes 8 *> out\paint\${run}_${ckpt}_validation_report.txt
}
"$(Get-Date -Format s) done $run" | Out-File -Append -Encoding utf8 out\paint\overnight_progress.txt
