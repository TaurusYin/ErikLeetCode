import sqlite3
import matplotlib.pyplot as plt
from collections import defaultdict
from datetime import datetime

# Connect to the SQLite database
conn = sqlite3.connect('online_learning.db')
cursor = conn.cursor()

# Aggregate the number of logins per month for each student
query = '''
SELECT 
    strftime('%Y-%m', login_time) as login_month, 
    count(student_id) as monthly_active_users 
FROM login_times 
GROUP BY strftime('%Y-%m', login_time), student_id
'''

cursor.execute(query)
results = cursor.fetchall()

# Count the number of monthly active users
counts = defaultdict(int)
for result in results:
    login_month, monthly_active_users = result
    counts[login_month] += monthly_active_users

# Plot the number of monthly active users
months = list(counts.keys())
mau = list(counts.values())
plt.plot(months, mau)
plt.xlabel('Month')
plt.ylabel('Monthly Active Users (MAU)')
plt.title('Change of Monthly Active Users (MAU)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Close the database connection
conn.close()
