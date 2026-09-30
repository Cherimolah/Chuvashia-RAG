"""Нарезка диалога на перекрывающиеся окна для эмбеддинга."""

from chuvashia_rag.logger import get_logger

log = get_logger(__name__)


def chunk_dialogue(messages, chunk_size=3, overlap=1):
    chunks = []
    if len(messages) < chunk_size:
        for i in range(0, len(messages)):
            text = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
            chunks.append({
                "text": text,
                "start_turn": i,
                "end_turn": i + len(messages) - 1
            })
        log.debug(
            f'chunk_dialogue: messages={len(messages)} < chunk_size={chunk_size}, '
            f'создал {len(chunks)} коротких чанков'
        )
        return chunks

    for i in range(0, len(messages), chunk_size - overlap):
        window = messages[i:i + chunk_size]
        if len(window) < 2:  # пропускаем слишком маленькие хвосты
            continue
        text = "\n".join(f"{m['role']}: {m['content']}" for m in window)
        chunks.append({
            "text": text,
            "start_turn": i,
            "end_turn": i + len(window) - 1
        })
    log.debug(
        f'chunk_dialogue: messages={len(messages)}, '
        f'chunk_size={chunk_size}, overlap={overlap} → {len(chunks)} чанков'
    )
    return chunks
