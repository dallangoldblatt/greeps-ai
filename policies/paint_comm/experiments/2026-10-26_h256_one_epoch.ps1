# Size comparison, retries after size3_h256_a1 (2026-10-26). a1 (lr 1e-4) did
# not go NaN but collapsed: policy entropy 1.13 -> 0.46, score 51 -> ~45, best
# checkpoint at update 25. With 16 minibatches x 4 epochs it takes 64 gradient
# steps per update against the 64-wide control's 16.
#   b1: 1 epoch -> 16 steps per update, the same as the control.
#   b2: 4 epochs at half the learning rate and a tighter gradient clip.
$ErrorActionPreference = "Continue"
Set-Location (Resolve-Path "$PSScriptRoot\..\..\..")   # repository root
$common = "--minutes 180 --init-from trained/paint_comm/checkpoints/night2_all_latest.pt --custom-maps maps/training " +
          "--select-maps maps/validation --custom-weight 0.7 --real-weight 0 --random-maps 128 " +
          "--colour-incentives --colour-diversity-coef 0.01 --incentive-floor 0.2 --anneal-updates 1 " +
          "--game-seconds 120 --game-seconds-final 120 --trainable-memory --memory-hardness-start 1 " +
          "--eval-episodes 4 --team-share 0.5 --bptt 64 " +
          "--colour-entropy-target 0.2 --memory-noise 0.02 --hidden 256 --blocks 2 --minibatches 16 " +
          "--colour-alpha-max 0.05"
$attempts = [ordered]@{
    "size3_h256_b1" = "--lr 1e-4 --epochs 1"
    "size3_h256_b2" = "--lr 5e-5 --max-grad-norm 0.25"
}
foreach ($run in $attempts.Keys) {
    "$(Get-Date -Format s) start $run ($($attempts[$run]))" | Out-File -Append -Encoding utf8 out\paint\overnight_progress.txt
    Invoke-Expression "python -m policies.paint_comm.train --run $run $common $($attempts[$run]) *> out\paint\${run}_console.txt"
    if (Select-String -Path out\paint\${run}_console.txt -Pattern "loss is not finite" -Quiet) {
        "$(Get-Date -Format s) $run diverged" | Out-File -Append -Encoding utf8 out\paint\overnight_progress.txt
        continue
    }
    foreach ($ckpt in "best", "latest") {
        "$(Get-Date -Format s) report $run $ckpt" | Out-File -Append -Encoding utf8 out\paint\overnight_progress.txt
        python -m tools.paint_report --policy paint_comm --checkpoint out/paint/$run/$ckpt.pt `
            --maps maps/validation --episodes 8 *> out\paint\${run}_${ckpt}_validation_report.txt
    }
}
"$(Get-Date -Format s) done size retries b1/b2" | Out-File -Append -Encoding utf8 out\paint\overnight_progress.txt
