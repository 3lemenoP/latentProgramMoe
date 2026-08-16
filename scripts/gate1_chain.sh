#!/bin/bash
# Gate-1 master chain — single L4 studio (gate1-steering.md, L4-only directive).
# Order: bridge (fast verdict) -> whitening -> G1-A -> G1-B -> G1-C.
cd /teamspace/studios/this_studio/latentProgramMoe
P=/home/zeus/miniconda3/envs/cloudspace/bin/python
set -e
echo "=== 0. quarter-rotary bridge ==="
$P scripts/b3_rotary_ladder.py --rotary-pct 0.25 --dir runs/bridge > g1_bridge.log 2>&1
echo "=== 1. pythia whitening from B2 ==="
$P scripts/b2_to_whitening.py --b2 runs/b2/report_b2.json --out runs/w0_pythia/whitening.json > g1_whitening.log 2>&1
echo "=== 2. G1-A teachers + program ==="
$P scripts/make_experts.py --base EleutherAI/pythia-410m --out experts-800 --tasks french --steps 800 > g1a_t800.log 2>&1
$P scripts/make_experts.py --base EleutherAI/pythia-410m --out experts-2000 --tasks french --steps 2000 > g1a_t2000.log 2>&1
$P scripts/fit_expert.py --config configs/b4_pythia.yaml --task french --experts-dir experts-800 --out runs/g1a --tag conj_only --freeze-sites rope_ax > g1a_fit.log 2>&1
$P scripts/g1a_analyze.py --teacher800 experts-800/french/merged --teacher2000 experts-2000/french/merged --program runs/g1a/z_french_conj_only.pt --out runs/g1a > g1a_analyze.log 2>&1
echo "=== 3. G1-B breadth ==="
for S in formal jsonish medical hedge; do
  $P scripts/make_experts.py --base EleutherAI/pythia-410m --out experts-2000 --tasks $S --steps 2000 > g1b_teacher_$S.log 2>&1
  $P scripts/fit_expert.py --config configs/b4_pythia.yaml --task $S --experts-dir experts-2000 --out runs/g1b --tag conj_only --freeze-sites rope_ax > g1b_fit_$S.log 2>&1
done
$P scripts/fit_expert.py --config configs/b4_pythia.yaml --task jsonish --experts-dir experts-2000 --out runs/g1b --tag conj_axis > g1b_fit_jsonish_axis.log 2>&1
$P scripts/g1b_analyze.py > g1b_analyze.log 2>&1
echo "=== 4. G1-C pruning ==="
$P scripts/g1c_prune.py --base EleutherAI/pythia-410m --program runs/g1a/z_french_conj_only.pt --task french --teacher experts-2000/french/merged --out runs/g1c > g1c.log 2>&1
echo "GATE1 CHAIN COMPLETE"
