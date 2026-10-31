from __future__ import annotations

import torch

from sepsis.train_dggru_formal import LANDMARK_INDEX, LandmarkModel


def test_landmark_outputs_do_not_use_future_sequence_values() -> None:
    torch.manual_seed(7)
    model = LandmarkModel(
        n_sequence=8,
        n_static=4,
        classes=3,
        head="nominal",
        hidden=16,
        dropout=0.0,
        use_graph=True,
    )
    model.eval()

    sequence = torch.randn(1, 72, 8)
    static = torch.randn(1, 4)
    altered_after_24 = sequence.clone()
    altered_after_24[:, LANDMARK_INDEX[0] + 1 :, :] = torch.randn(
        1, 72 - LANDMARK_INDEX[0] - 1, 8
    )
    altered_after_48 = sequence.clone()
    altered_after_48[:, LANDMARK_INDEX[1] + 1 :, :] = torch.randn(
        1, 72 - LANDMARK_INDEX[1] - 1, 8
    )

    with torch.no_grad():
        _, original = model(sequence, static)
        _, after_24 = model(altered_after_24, static)
        _, after_48 = model(altered_after_48, static)

    assert torch.allclose(original[:, :24], after_24[:, :24], atol=1e-6)
    assert torch.allclose(original[:, :48], after_48[:, :48], atol=1e-6)
