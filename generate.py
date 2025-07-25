import csv
import random

def generate_csv(filename, headers, target_averages, num_respondents, min_rating, max_rating, departments):
    respondents_data = []

    for _ in range(num_respondents):
        row = []
        for avg in target_averages:
            choices = list(range(min_rating, max_rating + 1))
            weights = [1 / (abs(val - avg) ** 2 + 0.1) for val in choices]
            score = random.choices(choices, weights=weights, k=1)[0]
            row.append(score)
        department = random.choice(departments)
        row.append(department)
        respondents_data.append(row)

    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(respondents_data)

    print(f"✅ '{filename}' with {num_respondents} respondents has been generated.")

# 📝 Headers
headers = [
    "Objectives - Contribution",
    "Objectives - Timeliness",
    "Objectives - Usefulness",
    "Objectives - Clarity",
    "OrgPrep - Planning",
    "OrgPrep - GroupPrep",
    "OrgPrep - Time",
    "OrgPrep - Venue",
    "OrgPrep - Zest",
    "Involvement - Enthusiasm",
    "Involvement - Level",
    "department_name"
]

# 🏫 Departments
departments = ["SAB", "BE", "SIT", "SHS","SAS","SED"]

# 🎯 Target averages per satisfaction level
levels = {
    "low":     [1.9, 2.0, 1.8, 2.1, 2.0, 1.7, 1.9, 2.0, 1.8, 2.1, 1.9],
    "average": [3.2, 3.3, 3.0, 3.1, 3.4, 3.2, 3.3, 3.2, 3.1, 3.3, 3.2],
    "high":    [4.9, 4.8, 4.9, 4.95, 4.85, 4.9, 4.8, 4.9, 4.85, 4.95, 4.9]
}

# 📅 Year, satisfaction level, and respondent count
year_configs = {
    2022: ("average", 1340),
    2023: ("average", 1263),
    2024: ("high", 1754)
}

# ⚙ Rating range
min_rating = 1
max_rating = 5

# 🔁 Generate each year's file
for year, (level, count) in year_configs.items():
    filename = f"spuqc_{year}.csv"
    generate_csv(
        filename=filename,
        headers=headers,
        target_averages=levels[level],
        num_respondents=count,
        min_rating=min_rating,
        max_rating=max_rating,
        departments=departments
    )
