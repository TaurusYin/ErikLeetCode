import sqlite3
import random
import names
import faker

conn = sqlite3.connect('online_learning.db')
cursor = conn.cursor()

fake = faker.Faker()

# Generate dummy data for course_category table
course_categories = [('Computer Science',), ('Mathematics',), ('Physics',), ('Chemistry',), ('Biology',),
                     ('Literature',), ('History',), ('Geography',), ('Music',), ('Art',), ('Sports',), ('Economics',),
                     ('Political Science',), ('Sociology',), ('Philosophy',), ('Psychology',), ('Education',),
                     ('Business',), ('Engineering',), ('Law',)]
cursor.executemany('INSERT INTO course_category (name) VALUES (?)', course_categories)

# Generate dummy data for course table
courses = []
for i in range(20):
    courses.append((f'{fake.job()} {i + 1}', random.choice(course_categories)[0]))
cursor.executemany('INSERT INTO course (name, category) VALUES (?, ?)', courses)

# Generate dummy data for student table
students = []
for i in range(1000):
    students.append((names.get_full_name(), random.choice(['Male', 'Female']), fake.country()))
cursor.executemany('INSERT INTO student (name, sex, country) VALUES (?, ?, ?)', students)

# Generate dummy data for login_times table
login_times = []
for i in range(10000):
    login_times.append((
        random.choice(students)[0],
        random.choice(courses)[0],
        fake.date_time_this_decade().strftime('%Y-%m-%d %H:%M:%S')
    ))
cursor.executemany('INSERT INTO login_times (student_id, course_id, login_time) VALUES (?, ?, ?)', login_times)

# Generate dummy data for quiz_scores table
quiz_scores = []
for i in range(10000):
    quiz_scores.append((
        random.choice(students)[0],
        random.choice(courses)[0],
        random.uniform(0, 100)
    ))
cursor.executemany('INSERT INTO quiz_scores (student_id, course_id, score) VALUES (?, ?, ?)', quiz_scores)

conn.commit()
conn.close()
