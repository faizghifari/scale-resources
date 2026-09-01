#!/usr/bin/env bash
set -euo pipefail

# RETIRED. This used to call `scripts.train_discriminator` and
# `scripts.compute_fted` to train a human-vs-synthetic discriminator and
# score Frechet embedding distance for Balinese/Cirebonese. Both modules'
# source was already gone from scripts/ before this repo's 2026-09 refactor
# (only stale .pyc bytecode remained), so that functionality was abandoned
# upstream of this change, not by it. Removed rather than reconstructed --
# see models/discriminator_balinese.pkl and models/discriminator_cirebonese.pkl
# for the last artifacts this pipeline produced, if you need to rebuild it.
