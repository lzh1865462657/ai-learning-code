import pytest

from ai_learning.python_basics import Student, summarize_students


def test_student_average_and_grade() -> None:
    student = Student("小林", (95, 90, 85))

    assert student.average == 90
    assert student.grade == "A"


def test_student_rejects_invalid_score() -> None:
    with pytest.raises(ValueError, match="0 到 100"):
        Student("小林", (101,))


def test_summarize_students() -> None:
    summary = summarize_students(
        [Student("甲", (90, 80)), Student("乙", (50, 60))]
    )

    assert summary["student_count"] == 2
    assert summary["class_average"] == 70
    assert summary["pass_rate"] == 0.5
    assert summary["top_student"]["name"] == "甲"
