"""tests/test_card.py — report-card generator tests TC-1..TC-3
(skill-report-card-spec §4). TC-4/TC-5 are studio-side assertions in the
G1-B analyzer (they need real fitted artifacts)."""
import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from lpm import LatentProgramModel, ProgramField  # noqa: E402
from lpm.report_card import build_card  # noqa: E402
from lpm.card_render import render_html, render_md  # noqa: E402

torch.manual_seed(0)


@pytest.fixture(scope="module")
def model():
    from e3_common import make_base
    from lpm.tasks import E3Vocab
    return LatentProgramModel(make_base(E3Vocab(), "cpu"))


@pytest.fixture(scope="module")
def blocks(model):
    g = torch.Generator().manual_seed(1)
    return torch.randint(0, model.config.vocab_size, (6, 24), generator=g)


def test_tc1_identity_card_green(model, blocks, tmp_path):
    field = model.identity_field()
    p = tmp_path / "z_id.pt"
    field.save(str(p))
    card = build_card(model=model, field=field, base_id="toy", device="cpu",
                      program_path=str(p), skill_name="identity",
                      task_eval_blocks=None, neutral_blocks=blocks,
                      provenance={"note": "TC-1"})
    assert card["fidelity"] == "N/A"
    assert card["all_certificates_pass"]
    assert card["certificates"]["identity_check"]["pass"]
    assert card["certificates"]["undo_attestation"]["pass"]
    md = render_md(card)
    assert "identity" in md and "Certificates" in md
    assert "<body" in render_html(card)


def test_tc2_tampered_program_fails(model, blocks, tmp_path):
    field = ProgramField.randn_near_identity(model.spec, sigma=0.02)
    k = model.spec.site_keys()[3]
    q = field.q(*k)
    with torch.no_grad():
        q.reshape(-1, 4)[0] *= 10.0        # non-unit-band quaternion
    p = tmp_path / "z_bad.pt"
    field.save(str(p))
    card = build_card(model=model, field=field, base_id="toy", device="cpu",
                      program_path=str(p), skill_name="tampered",
                      neutral_blocks=blocks)
    assert not card["certificates"]["spectrum"]["pass"]
    assert not card["all_certificates_pass"]


def test_tc3_determinism(model, blocks, tmp_path):
    field = ProgramField.randn_near_identity(model.spec, sigma=0.02)
    p = tmp_path / "z_det.pt"
    field.save(str(p))
    kw = dict(model=model, field=field, base_id="toy", device="cpu",
              program_path=str(p), skill_name="det", neutral_blocks=blocks,
              provenance={"fixed": True}, seed=7)
    a = json.dumps(build_card(**kw), sort_keys=True)
    b = json.dumps(build_card(**kw), sort_keys=True)
    assert a == b


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
