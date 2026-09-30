import pytest

pytest.importorskip("ragas", reason="нужна группа eval: poetry install --with eval")

from chuvashia_rag.evaluation.ragas_evaluation import NOISE_DOCUMENTS, inject_noise  # noqa: E402


def test_inject_noise_keeps_originals_and_adds_noise():
    contexts = ["Акатуй — летний праздник чувашей", "Сурхури — зимний праздник"]

    noisy = inject_noise(contexts, noise_ratio=0.5, seed=42)

    assert all(c in noisy for c in contexts)
    added = [c for c in noisy if c not in contexts]
    assert len(added) == 2
    assert all(c in NOISE_DOCUMENTS for c in added)


def test_inject_noise_is_reproducible_with_seed():
    contexts = ["a", "b", "c"]

    assert inject_noise(contexts, 0.3, seed=1) == inject_noise(contexts, 0.3, seed=1)


def test_inject_noise_does_not_mutate_input():
    contexts = ["a"]

    inject_noise(contexts, 0.5, seed=0)

    assert contexts == ["a"]
