"""The ablation ladder (deck slides 10-11, 43).

Each rung adds exactly one component, so a delta in the table attributes to
exactly one design decision. That is the whole point -- a table where two things
change at once diagnoses nothing.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    name: str
    sparse: bool
    dense: bool
    rerank: bool
    hyde: bool

    def __post_init__(self) -> None:
        if not (self.sparse or self.dense):
            raise ValueError(f"{self.name}: needs at least one retrieval arm")


ABLATION_LADDER: list[Config] = [
    Config("naive",    sparse=False, dense=True, rerank=False, hyde=False),
    Config("+hybrid",  sparse=True,  dense=True, rerank=False, hyde=False),
    Config("+rerank",  sparse=True,  dense=True, rerank=True,  hyde=False),
    Config("+hyde",    sparse=True,  dense=True, rerank=True,  hyde=True),
]


def to_markdown(rows: list[dict]) -> str:
    if not rows:
        return ""
    cols = list(rows[0].keys())
    expected_keys = set(cols)

    # Validate that every row has exactly the expected columns
    for i, r in enumerate(rows):
        row_keys = set(r.keys())
        if row_keys != expected_keys:
            missing = expected_keys - row_keys
            extra = row_keys - expected_keys
            if missing and extra:
                raise ValueError(
                    f"row {i} has keys {row_keys}; expected {expected_keys} "
                    f"(missing: {missing}, extra: {extra})"
                )
            elif missing:
                raise ValueError(
                    f"row {i} has keys {row_keys}; expected {expected_keys} "
                    f"(missing: {missing})"
                )
            else:  # extra
                raise ValueError(
                    f"row {i} has keys {row_keys}; expected {expected_keys} "
                    f"(extra: {extra})"
                )

    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = [
        "| " + " | ".join(
            f"{r[c]:.4f}" if isinstance(r[c], float) else str(r[c]) for c in cols
        ) + " |"
        for r in rows
    ]
    return "\n".join([head, sep, *body])
