"""Schema round-trip, stable_id determinism, and JSONL resume semantics."""

import json

from jeff.schema import (
    Generation,
    RouterRow,
    ScoredRow,
    Task,
    completed_ids,
    read_jsonl,
    stable_id,
    write_jsonl,
)


def _task(i: int = 0) -> Task:
    return Task(
        id=f"task_{i:04d}",
        source="gsm8k",
        task_family="math",
        prompt=f"What is {i} + {i}?",
        reference=str(2 * i),
        grader="math_numeric",
        metadata={"group_id": f"g{i % 3}"},
    )


def test_task_jsonl_round_trip(tmp_path):
    path = tmp_path / "tasks.jsonl"
    tasks = [_task(i) for i in range(5)]
    assert write_jsonl(path, tasks) == 5
    back = read_jsonl(path, Task)
    assert back == tasks
    assert back[0].group_id() == "g0"


def test_write_jsonl_appends(tmp_path):
    path = tmp_path / "gen.jsonl"
    g = Generation(task_id="t1", model="m", answer="a")
    write_jsonl(path, [g])
    write_jsonl(path, [g])
    assert len(read_jsonl(path, Generation)) == 2


def test_scored_and_router_row_round_trip(tmp_path):
    row = ScoredRow(
        id="r1", source="mbpp", task_family="code", prompt="p",
        local_model="e4b", frontier_model="fr",
        local_score=0.5, frontier_score=0.9, delta_q=0.4,
    )
    path = tmp_path / "scored.jsonl"
    write_jsonl(path, [row])
    assert read_jsonl(path, ScoredRow)[0].delta_q == 0.4

    rr = RouterRow(
        id="r1", source="mbpp", task_family="code", prompt="p",
        group_id="r1", split="train",
        local_score=0.5, frontier_score=0.9, delta_q=0.4,
        local_sufficient=False, frontier_helpful=True,
    )
    path2 = tmp_path / "router.jsonl"
    write_jsonl(path2, [rr])
    assert read_jsonl(path2, RouterRow)[0].frontier_helpful is True


def test_stable_id_deterministic():
    a = stable_id("gsm8k", "prompt text")
    b = stable_id("gsm8k", "prompt text")
    c = stable_id("gsm8k", "other prompt")
    assert a == b and a != c
    assert len(a) == 16
    assert len(stable_id("x", n=8)) == 8
    # None parts are treated as empty strings.
    assert stable_id(None, "x") == stable_id("", "x")


def test_completed_ids_resume(tmp_path):
    path = tmp_path / "out.jsonl"
    assert completed_ids(path) == set()  # missing file -> empty

    gens = [Generation(task_id=f"t{i}", model="m", answer="a") for i in range(3)]
    write_jsonl(path, gens)
    assert completed_ids(path) == {"t0", "t1", "t2"}

    # Malformed lines are skipped, not fatal.
    with open(path, "a", encoding="utf-8") as f:
        f.write("not json\n")
        f.write(json.dumps({"other_key": 1}) + "\n")
    assert completed_ids(path) == {"t0", "t1", "t2"}
    assert completed_ids(path, key="other_key") == {1}
