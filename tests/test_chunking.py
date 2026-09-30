from chuvashia_rag.chunking import chunk_dialogue


def _dialogue(n: int) -> list[dict[str, str]]:
    return [{"role": "user" if i % 2 == 0 else "assistant", "content": f"msg{i}"} for i in range(n)]


def test_single_message_gives_one_chunk_with_role_prefix():
    chunks = chunk_dialogue(_dialogue(1))

    assert len(chunks) == 1
    assert chunks[0]["text"] == "user: msg0"


def test_short_dialogue_repeats_whole_dialogue_in_every_chunk():
    # Текущее поведение: при len < chunk_size каждый чанк — весь диалог целиком.
    # Для mean pooling результат тот же, но это лишние токены (см. docs/TECH_DEBT.md).
    chunks = chunk_dialogue(_dialogue(2), chunk_size=3)

    assert len(chunks) == 2
    assert all(c["text"] == "user: msg0\nassistant: msg1" for c in chunks)


def test_long_dialogue_uses_overlapping_windows():
    chunks = chunk_dialogue(_dialogue(5), chunk_size=3, overlap=1)

    assert [(c["start_turn"], c["end_turn"]) for c in chunks] == [(0, 2), (2, 4)]
    assert chunks[1]["text"].startswith("user: msg2")


def test_single_message_tail_is_skipped():
    # окно из одного сообщения в хвосте отбрасывается
    chunks = chunk_dialogue(_dialogue(4), chunk_size=3, overlap=1)

    assert [(c["start_turn"], c["end_turn"]) for c in chunks] == [(0, 2), (2, 3)]
