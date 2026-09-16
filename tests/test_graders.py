"""Grader behavior on crafted pass/fail cases; garbage must never raise."""

import pytest

from jeff.graders import GRADERS, GradeResult, grade
from jeff.schema import Task


def _task(grader: str, reference=None, **metadata) -> Task:
    return Task(
        id="t", source="test", task_family="test", prompt="p",
        reference=reference, grader=grader, metadata=metadata,
    )


def _check(res: GradeResult) -> None:
    assert isinstance(res, GradeResult)
    assert 0.0 <= res.score <= 1.0
    assert res.grader_type == "deterministic"
    assert res.confidence == 1.0


# --- exact_match -----------------------------------------------------------


def test_exact_match_normalizes_case_and_whitespace():
    t = _task("exact_match", reference="The Answer  Is 42")
    assert grade(t, "the answer is 42").score == 1.0
    assert grade(t, "  THE   ANSWER IS 42 \n").score == 1.0
    assert grade(t, "something else").score == 0.0


# --- math_numeric ----------------------------------------------------------


def test_math_numeric_formats():
    t = _task("math_numeric", reference="42")
    assert grade(t, "blah blah\n#### 42").score == 1.0
    assert grade(t, "so \\boxed{42}").score == 1.0
    assert grade(t, "the answer is 42").score == 1.0
    assert grade(t, "the answer is $42.00").score == 1.0
    assert grade(t, "it is 1,000 then finally 42").score == 1.0  # last number
    assert grade(t, "#### 41").score == 0.0


def test_math_numeric_fractions_and_tolerance():
    assert grade(_task("math_numeric", reference="0.75"), "\\boxed{3/4}").score == 1.0
    assert grade(_task("math_numeric", reference="3/4"), "0.75").score == 1.0
    # rel tolerance 1e-4 (math.isclose is inclusive at the boundary)
    assert grade(_task("math_numeric", reference="10000"), "10002").score == 0.0
    assert grade(_task("math_numeric", reference="10000"), "10000.5").score == 1.0


def test_math_numeric_garbage():
    res = grade(_task("math_numeric", reference="42"), "no numbers here")
    _check(res)
    assert res.score == 0.0


# --- multiple_choice -------------------------------------------------------


def test_multiple_choice_extraction():
    t = _task("multiple_choice", answer_letter="B")
    assert grade(t, "(B)").score == 1.0
    assert grade(t, "B.").score == 1.0
    assert grade(t, "answer: B").score == 1.0
    assert grade(t, "the answer is B").score == 1.0
    assert grade(t, "\\boxed{B}").score == 1.0
    assert grade(t, "B").score == 1.0
    assert grade(t, "(C)").score == 0.0


def test_multiple_choice_answer_index():
    t = _task("multiple_choice", answer_index=2)  # -> C
    assert grade(t, "(C)").score == 1.0
    assert grade(t, "(A)").score == 0.0


# --- code_tests ------------------------------------------------------------

PASS_TASK = _task(
    "code_tests",
    test_code="assert add(2, 3) == 5\nassert add(-1, 1) == 0",
)


def test_code_tests_pass():
    res = grade(PASS_TASK, "```python\ndef add(a, b):\n    return a + b\n```")
    _check(res)
    assert res.score == 1.0


def test_code_tests_fail_assert_and_syntax():
    assert grade(PASS_TASK, "```python\ndef add(a, b):\n    return a - b\n```").score == 0.0
    res = grade(PASS_TASK, "```python\ndef add(:\n```")
    assert res.score == 0.0
    assert res.detail.get("returncode") != 0 or res.detail.get("error")


def test_code_tests_raw_answer_and_garbage():
    assert grade(PASS_TASK, "def add(a, b):\n    return a + b").score == 1.0
    assert grade(PASS_TASK, "").score == 0.0
    assert grade(PASS_TASK, "I cannot write code").score == 0.0


# --- json_schema -----------------------------------------------------------

SCHEMA = {
    "type": "object",
    "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
    "required": ["name"],
}


def test_json_schema():
    t = _task("json_schema", schema=SCHEMA)
    assert grade(t, '{"name": "al", "age": 3}').score == 1.0
    assert grade(t, '```json\n{"name": "al"}\n```').score == 1.0
    assert grade(t, '{"age": 3}').score == 0.0  # missing required
    assert grade(t, '{"name": "al", "age": "x"}').score == 0.0
    assert grade(t, "not json at all").score == 0.0


# --- ifeval ----------------------------------------------------------------


def test_ifeval_all_satisfied():
    t = _task(
        "ifeval",
        constraints=[
            {"instruction_id": "keywords:existence", "kwargs": {"keywords": ["apple"]}},
            {"instruction_id": "punctuation:no_comma", "kwargs": {}},
            {"instruction_id": "length_constraints:number_words",
             "kwargs": {"num_words": 3, "relation": "at least"}},
        ],
    )
    res = grade(t, "an apple a day")
    assert res.score == 1.0
    assert res.detail["satisfied"] == 3


def test_ifeval_partial_and_unknown():
    t = _task(
        "ifeval",
        constraints=[
            {"instruction_id": "keywords:existence", "kwargs": {"keywords": ["apple"]}},
            {"instruction_id": "punctuation:no_comma", "kwargs": {}},
            {"instruction_id": "madeup:unknown_thing", "kwargs": {}},
        ],
    )
    res = grade(t, "an apple, a day")  # comma fails no_comma
    assert res.score == pytest.approx(0.5)
    assert res.detail["unknown"] == ["madeup:unknown_thing"]
    assert res.detail["checked"] == 2


def test_ifeval_verifier_subset():
    cases = [
        ("change_case:english_capital", {}, "ALL CAPS HERE", True),
        ("change_case:english_lowercase", {}, "all lower", True),
        ("startend:end_checker", {"end_phrase": "The end."}, "blah The end.", True),
        ("startend:quotation", {}, '"quoted text"', True),
        ("detectable_format:title", {}, "text <<My Title>> more", True),
        ("detectable_format:number_bullet_lists", {"num_bullets": 2},
         "* one\n* two", True),
        ("detectable_format:json_format", {}, '{"a": 1}', True),
        ("detectable_format:constrained_response", {}, "My answer is yes.", True),
        ("detectable_format:number_highlighted_sections", {"num_highlights": 2},
         "*a* and **b**", True),
        ("detectable_format:multiple_sections",
         {"section_spliter": "Section", "num_sections": 2},
         "Section 1 x\nSection 2 y", True),
        ("keywords:forbidden_words", {"forbidden_words": ["bad"]}, "all good", True),
        ("keywords:frequency",
         {"keyword": "cat", "frequency": 2, "relation": "at least"},
         "cat cat cat", True),
        ("length_constraints:num_sentences",
         {"num_sentences": 2, "relation": "at least"}, "One. Two. Three.", True),
        ("length_constraints:num_paragraphs", {"num_paragraphs": 2},
         "para one\n\npara two", True),
        ("combination:two_responses", {}, "resp one\n******\nresp two", True),
        ("combination:repeat_prompt", {"prompt_to_repeat": "say hi"},
         "say hi to everyone", True),
        ("change_case:capital_word_frequency",
         {"capital_frequency": 0.5, "capital_relation": "at least"},
         "THIS IS half CAPS", True),
    ]
    # forbidden_words is vacuously satisfied by an empty answer (correct),
    # so it gets a different negative control.
    vacuous_ok = {"keywords:forbidden_words"}
    for iid, kwargs, answer, expected in cases:
        t = _task("ifeval", constraints=[{"instruction_id": iid, "kwargs": kwargs}])
        res = grade(t, answer)
        assert res.score == (1.0 if expected else 0.0), f"{iid} on {answer!r}"
        # negative control: same constraint should fail on empty answer
        res_empty = grade(t, "")
        if iid in vacuous_ok:
            assert res_empty.score == 1.0
        else:
            assert res_empty.score == 0.0, f"{iid} should fail on empty answer"


def test_ifeval_no_constraints_and_garbage():
    assert grade(_task("ifeval"), "anything").score == 1.0
    res = grade(_task("ifeval", constraints="not-a-list"), "x")
    _check(res)


# --- tool_call -------------------------------------------------------------

EXPECTED = {"name": "get_weather", "arguments": {"city": "Paris", "units": "celsius"}}


def test_tool_call_scoring():
    t = _task("tool_call", expected=EXPECTED)
    full = grade(t, '{"name": "get_weather", "arguments": {"city": "Paris", "units": "celsius"}}')
    assert full.score == 1.0
    name_only = grade(t, '{"name": "get_weather", "arguments": {"city": "London", "units": "f"}}')
    assert name_only.score == pytest.approx(0.5)
    wrong_name = grade(t, '{"name": "other", "arguments": {"city": "Paris", "units": "celsius"}}')
    assert wrong_name.score == pytest.approx(0.5)
    nothing = grade(t, '{"name": "other", "arguments": {}}')
    assert nothing.score == 0.0
    assert grade(t, "no json").score == 0.0


# --- reference_similarity --------------------------------------------------


def test_reference_similarity():
    t = _task("reference_similarity", reference="the cat sat on the mat")
    assert grade(t, "the cat sat on the mat").score == 1.0
    partial = grade(t, "the cat sat")
    assert 0.0 < partial.score < 1.0
    assert grade(t, "completely different words").score == 0.0


# --- dispatch --------------------------------------------------------------


def test_llm_judge_not_implemented():
    with pytest.raises(NotImplementedError):
        grade(_task("llm_judge"), "answer")


def test_registry_covers_all_grader_names():
    from jeff.schema import GraderName
    import typing

    assert set(GRADERS) == set(typing.get_args(GraderName))


@pytest.mark.parametrize("grader_name", sorted(set(GRADERS) - {"llm_judge"}))
def test_graders_never_raise_on_garbage(grader_name):
    t = _task(grader_name, reference="ref")
    for garbage in ("", "   ", "\x00\xff garbage", "{}", "[]", "```\n```", "9" * 10_000):
        res = grade(t, garbage)
        _check(res)
