from pprint import pprint

from ai_learning.python_basics import Student, summarize_students

students = [
    Student("小林", (92, 88, 95)),
    Student("小周", (76, 81, 79)),
    Student("小陈", (58, 64, 61)),
]

pprint(summarize_students(students), sort_dicts=False)
