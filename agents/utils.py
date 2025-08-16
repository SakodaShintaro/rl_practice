"""Utility functions for agent implementations."""


def update_and_pad_history(history, new_item, seq_len):
    """Update history with new item and pad to seq_len.

    Args:
        history: List to update (modified in-place)
        new_item: New item to append
        seq_len: Target sequence length
    """
    history.append(new_item)
    if len(history) > seq_len:
        history[:] = history[-seq_len:]

    # Pad to seq_len
    while len(history) < seq_len:
        history.insert(0, history[0])
