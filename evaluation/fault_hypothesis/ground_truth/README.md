# Fault hypothesis ground truth

The current labeled cases live in `../manifest.json` and are marked `development_synthetic`: each expected fault is an injected, known SPICE scenario with a simulation-derived observation. No independent repair or real-board fault labels are available yet. These cases support deterministic pipeline smoke checks only; do not use their scores as real-world fault accuracy.
