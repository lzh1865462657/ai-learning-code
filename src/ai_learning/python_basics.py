"""Python basics demonstrated through a small student score domain."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from statistics import mean


@dataclass(frozen=True, slots=True)
class Student:
    """A student and their scores for one or more courses."""

    name: str
    scores: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("学生姓名不能为空")
        if not self.scores:
            raise ValueError("至少需要一门课程成绩")
        if any(score < 0 or score > 100 for score in self.scores):
            raise ValueError("成绩必须在 0 到 100 之间")

    @property
    def average(self) -> float:
        return round(mean(self.scores), 2)

    @property
    def grade(self) -> str:
        average = self.average
        if average >= 90:
            return "A"
        if average >= 80:
            return "B"
        if average >= 70:
            return "C"
        if average >= 60:
            return "D"
        return "F"


def summarize_students(students: Iterable[Student]) -> dict[str, object]:
    """Return class statistics while preserving the best student's details."""

    student_list = list(students)
    if not student_list:
        raise ValueError("学生列表不能为空")

    top_student = max(student_list, key=lambda student: student.average)
    passed = sum(student.average >= 60 for student in student_list)
    return {
        "student_count": len(student_list),
        "class_average": round(mean(student.average for student in student_list), 2),
        "pass_rate": round(passed / len(student_list), 2),
        "top_student": {
            "name": top_student.name,
            "average": top_student.average,
            "grade": top_student.grade,
        },
    }
