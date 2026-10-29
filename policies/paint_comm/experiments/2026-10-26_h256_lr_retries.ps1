# Size comparison, third try (2026-10-26): 256 wide + 2 residual blocks, same
# recipe and start as size2_h64. The first try at lr 3e-4 diverged at update
# 117, so each attempt is more conservative; an attempt that diverges (train.py
# stops on a non-finite loss) is followed straight away by the next one.
# Selection on maps/validation.
$ErrorActionPreference = "Continue"
Set-Location (Resolve-Path "$PSScriptRoot\..\..\..")   # repository root
$common = "--minutes 180 --init-from trained/paint_comm/checkpoints/night2_all_latest.pt --custom-maps maps/training " +
          "--select-maps maps/validation --custom-weight 0.7 --real-weight 0 --random-maps 128 " +
          "--colour-incentives --colour-diversity-coef 0.01 --incentive-floor 0.2 --anneal-updates 1 " +
          "--game-seconds 120 --game-seconds-final 120 --trainable-memory --memory-hardness-start 1 " +
          "--eval-episodes 4 --team-share 0.5 --bptt 64 " +
          "--colour-entropy-target 0.2 --memory-noise 0.02 --hidden 256 --blocks 2 --minibatches 16"
$attempts = [ordered]@{
    "size3_h256_a1" = "--lr 1e-4 --colour-alpha-max 0.05"
    "size3_h256_a2" = "--lr 5e-5 --colour-alpha-max 0.05 --max-grad-norm 0.25"
    "size3_h256_a3" = "--lr 3e-5 --colour-alpha-max 0.02 --max-grad-norm 0.25 --clip 0.1"
    "size3_h256_a4" = "--lr 2e-5 --colour-alpha-max 0.02 --max-grad-norm 0.1 --clip 0.1 --epochs 2"
}
$done = $null
foreach ($run in $attempts.Keys) {
    "$(Get-Date -Format s) start $run ($($attempts[$run]))" | Out-File -Append -Encoding utf8 out\paint\overnight_progress.txt
    Invoke-Expression "python -m policies.paint_comm.train --run $run $common $($attempts[$run]) *> out\paint\${run}_console.txt"
    if (Select-String -Path out\paint\${run}_console.txt -Pattern "loss is not finite" -Quiet) {
        "$(Get-Date -Format s) $run diverged" | Out-File -Append -Encoding utf8 out\paint\overnight_progress.txt
        continue
    }
    $done = $run
    break
}
if ($done) {
    foreach ($ckpt in "best", "latest") {
        "$(Get-Date -Format s) report $done $ckpt" | Out-File -Append -Encoding utf8 out\paint\overnight_progress.txt
        python -m tools.paint_report --policy paint_comm --checkpoint out/paint/$done/$ckpt.pt `
            --maps maps/validation --episodes 8 *> out\paint\${done}_${ckpt}_validation_report.txt
    }
}
"$(Get-Date -Format s) done size retry (finished: $done)" | Out-File -Append -Encoding utf8 out\paint\overnight_progress.txt
