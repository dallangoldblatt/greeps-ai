# Renders the animations in the README (tools/render_episode.py). Each shows
# the median of 8 episodes on the GPU simulator. About 15 minutes.
Set-Location (Resolve-Path "$PSScriptRoot\..\..")
$memory_st = "trained/paint_comm/checkpoints/memory_st_latest.pt"
$colour = "trained/paint_comm/checkpoints/colour_stage2_best.pt"
$val = "maps/validation/generated"

# Top of the README: trainable-memory policy, game lengthened to 360 s.
python -m tools.render_episode --policy paint_comm --checkpoint $memory_st `
    --map "$val/terrain_05_v008.map" --seconds 360 --every 8 --fps 18 --quality 50 `
    --out docs/media/terrain_05_v008.webp

# Greenfoot next to what the greeps sense, at half the speed of the others.
python -m tools.render_episode --policy paint_comm --checkpoint $colour `
    --map "$val/variety_06_v003.map" --view both --every 3 --fps 16 --quality 45 `
    --out docs/media/variety_06_v003_senses.webp

# Gallery: Map 1 without paint, two validation maps with paint (real 120 s games).
python -m tools.render_episode --policy linear_cma --map real:1 `
    --every 5 --fps 16 --scale 0.4 --quality 50 --out docs/media/map1_linear_cma.webp
python -m tools.render_episode --policy paint_comm --checkpoint $colour `
    --map "$val/puzzle_07_v018.map" --every 5 --fps 16 --scale 0.4 --quality 50 `
    --out docs/media/puzzle_07_v018.webp
python -m tools.render_episode --policy paint_comm --checkpoint $colour `
    --map "$val/signal_05_v009.map" --every 5 --fps 16 --scale 0.4 --quality 50 `
    --out docs/media/signal_05_v009.webp
