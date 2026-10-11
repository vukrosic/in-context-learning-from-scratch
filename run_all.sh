#!/bin/sh
# Runs every step in order. Trained networks are kept in nets/, so a second run is fast.
set -e
cd "$(dirname "$0")"
python3 tests.py
for f in step1_shop step2_weights step3_prompt step4_examples step5_ideal step6_where step7_diversity step8_limits; do
  echo; echo "== $f"; python3 $f.py
done
