# Size comparison (2026-10-23): the "all" recipe from night2_all/latest.pt for
# 3 hours each, 64-wide control vs 256 wide + 2 residual blocks (grown from the
# same actor, so both start with the same behaviour). Selection on
# maps/validation.
$ErrorActionPreference = "Continue"
Set-Location (Resolve-Path "$PSScriptRoot\..\..\..")   # repository root
$common = "--minutes 180 --init-from trained/paint_comm/checkpoints/night2_all_latest.pt --custom-maps maps/training " +
          "--select-maps maps/validation --custom-weight 0.7 --real-weight 0 --random-maps 128 " +
          "--colour-incentives --colour-diversity-coef 0.01 --incentive-floor 0.2 --anneal-updates 1 " +
          "--game-seconds 120 --game-seconds-final 120 --trainable-memory --memory-hardness-start 1 " +
          "--eval-episodes 4 --team-share 0.5 --bptt 64 --minibatches 4 " +
          "--colour-entropy-target 0.2 --memory-noise 0.02"
$arms = [ordered]@{
    "size2_h64"    = "--hidden 64 --blocks 0"
    "size2_h256b2" = "--hidden 256 --blocks 2"
}
foreach ($run in $arms.Keys) {
    "$(Get-Date -Format s) start $run" | Out-File -Append -Encoding utf8 out\paint\overnight_progress.txt
    Invoke-Expression "python -m policies.paint_comm.train --run $run $common $($arms[$run]) *> out\paint\${run}_console.txt"
}
foreach ($run in $arms.Keys) {
    foreach ($ckpt in "best", "latest") {
        "$(Get-Date -Format s) report $run $ckpt" | Out-File -Append -Encoding utf8 out\paint\overnight_progress.txt
        python -m tools.paint_report --policy paint_comm --checkpoint out/paint/$run/$ckpt.pt `
            --maps maps/validation --episodes 8 *> out\paint\${run}_${ckpt}_validation_report.txt
    }
}
"$(Get-Date -Format s) done size comparison" | Out-File -Append -Encoding utf8 out\paint\overnight_progress.txt
