"""
+---------------+      +---------------+      +---------------+
|    Student    |------| Login_Times   |      | Quiz_Scores   |
+---------------+      +---------------+      +---------------+
| PK student_id |      | login_id      |      | score_id      |
+---------------+      | FK student_id  |------| FK student_id |
| name          |      | login_timestamp|      | FK course_id  |
+---------------+      +---------------+      | score         |
| sex           |                              +---------------+
+---------------+
| country       |
+---------------+

+---------------+      +---------------+
|    Course     |------| Category      |
+---------------+      +---------------+
| PK course_id  |      | PK category_id|
+---------------+      | category_name |
| course_name   |      +---------------+
+---------------+      | FK course_category|
| FK course_category|------+---------------+
+---------------+

"""

import sqlite3

conn = sqlite3.connect('online_learning.db')
cursor = conn.cursor()

cursor.execute('''
CREATE TABLE IF NOT EXISTS student (
    student_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    sex TEXT NOT NULL,
    country TEXT NOT NULL
)
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS course (
    course_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    category TEXT NOT NULL
)
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS course_category (
    category_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL
)
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS login_times (
    id INTEGER PRIMARY KEY,
    student_id INTEGER NOT NULL,
    course_id INTEGER NOT NULL,
    login_time TEXT NOT NULL,
    FOREIGN KEY (student_id) REFERENCES student (student_id),
    FOREIGN KEY (course_id) REFERENCES course (course_id)
)
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS quiz_scores (
    id INTEGER PRIMARY KEY,
    student_id INTEGER NOT NULL,
    course_id INTEGER NOT NULL,
    score REAL NOT NULL,
    FOREIGN KEY (student_id) REFERENCES student (student_id),
    FOREIGN KEY (course_id) REFERENCES course (course_id)
)
''')

conn.commit()
conn.close()
