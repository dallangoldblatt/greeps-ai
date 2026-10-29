# Size comparison, second try (2026-10-23): 256 wide + 2 residual blocks with
# 16 minibatches (4 overflowed GPU memory), same recipe and start as the
# size2_h64 control. Selection on maps/validation.
$ErrorActionPreference = "Continue"
Set-Location (Resolve-Path "$PSScriptRoot\..\..\..")   # repository root
$run = "size2_h256b2_mb16"
"$(Get-Date -Format s) start $run" | Out-File -Append -Encoding utf8 out\paint\overnight_progress.txt
python -m policies.paint_comm.train --run $run --minutes 180 --init-from trained/paint_comm/checkpoints/night2_all_latest.pt `
    --custom-maps maps/training --select-maps maps/validation --custom-weight 0.7 --real-weight 0 `
    --random-maps 128 --colour-incentives --colour-diversity-coef 0.01 --incentive-floor 0.2 `
    --anneal-updates 1 --game-seconds 120 --game-seconds-final 120 --trainable-memory `
    --memory-hardness-start 1 --eval-episodes 4 --team-share 0.5 --bptt 64 `
    --colour-entropy-target 0.2 --memory-noise 0.02 --hidden 256 --blocks 2 --minibatches 16 `
    *> out\paint\${run}_console.txt
foreach ($r in "size2_h64", $run) {
    foreach ($ckpt in "best", "latest") {
        "$(Get-Date -Format s) report $r $ckpt" | Out-File -Append -Encoding utf8 out\paint\overnight_progress.txt
        python -m tools.paint_report --policy paint_comm --checkpoint out/paint/$r/$ckpt.pt `
            --maps maps/validation --episodes 8 *> out\paint\${r}_${ckpt}_validation_report.txt
    }
}
"$(Get-Date -Format s) done size comparison" | Out-File -Append -Encoding utf8 out\paint\overnight_progress.txt
