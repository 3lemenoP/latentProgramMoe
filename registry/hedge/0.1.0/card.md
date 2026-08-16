# Skill card — hedge v0.1.0

Base `EleutherAI/pythia-410m` · program `75915e1794aa46eb` · 59112 quats · 461.8 KB fp16 · 386.6 KB 8-bit sparse (56546 active sites)

## Fidelity
- task CE: base 3.4963 / program 2.9210 / teacher 2.8477
- recovery: **88.7%** (95% CI [84.9, 92.4]%, n=64)

## Gentleness
- neutral ppl: base 29.14 / program 25.14 / teacher 24.46
- collateral ratio: -147771060.467

## Certificates
- spectrum: PASS
- lipschitz: program-independent bound by construction (rotations are isometries; see tests T1/T6)
- identity_check: PASS
- undo_attestation: PASS

## Commitment (behavioral work shares)
| group | work share | raw activity (diag) |
|---|---:|---:|
| qk_rel | 0.463 | 0.00626 |
| ffn_hidden | 0.341 | 0.00551 |
| attn_io | 0.140 | 0.00628 |
| mlp_io | 0.055 | 0.00247 |
| rope_ax | 0.000 | 0.00000 |

Actuator class: **content**

## Activity
- mean 0.00512, max 0.5264

## Fingerprint
- probe losses: task:hedge 2.921, neutral:wikitext 3.224
- nearest: z_formal_conj_only (0.903), z_medical_conj_only (1.450), z_french_conj_only (1.476)

## Limits
This card certifies stability, size, scope of modification, and reversibility of the program. It does NOT certify content alignment, factuality, or safety of outputs — bounded modification class is not benign behavior. Unvalidated verbs: multi-skill stacking (composition of programs is retired); validated verbs: select, slerp, swap, undo.

## Provenance
```json
{
 "teacher": "experts-best/hedge/merged",
 "steps": 2000,
 "protocol": "G1-B budget-matched"
}
```