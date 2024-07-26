from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, count, lit, round
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

spark = SparkSession.builder \
    .appName("Projet Hadoop") \
    .config("spark.hadoop.fs.defaultFS", "hdfs://localhost:9000") \
    .getOrCreate()

df = spark.read.csv("hdfs://localhost:9000/data.csv", header=True, inferSchema=True)

print("describe:")
df.describe().show()
print("Types de données:")
df.printSchema()

df.show(5)

df.createOrReplaceTempView("data_view")

interview_score_avg = spark.sql("SELECT HiringDecision, AVG(InterviewScore) AS AvgInterviewScore FROM data_view GROUP BY HiringDecision").toPandas()
skill_score_avg = spark.sql("SELECT HiringDecision, AVG(SkillScore) AS AvgSkillScore FROM data_view GROUP BY HiringDecision").toPandas()
personality_score_avg = spark.sql("SELECT HiringDecision, AVG(PersonalityScore) AS AvgPersonalityScore FROM data_view GROUP BY HiringDecision").toPandas()
age_avg = spark.sql("SELECT HiringDecision, AVG(Age) AS AvgAge FROM data_view GROUP BY HiringDecision").toPandas()
experience_years_avg = spark.sql("SELECT HiringDecision, AVG(ExperienceYears) AS AvgExperienceYears FROM data_view GROUP BY HiringDecision").toPandas()
education_count = spark.sql("SELECT EducationLevel, HiringDecision, COUNT(*) AS Count FROM data_view GROUP BY EducationLevel, HiringDecision").toPandas()
gender_count = spark.sql("SELECT Gender, HiringDecision, COUNT(*) AS Count FROM data_view GROUP BY Gender, HiringDecision").toPandas()
distance_avg = spark.sql("SELECT HiringDecision, AVG(ROUND(DistanceFromCompany, 2)) AS AvgDistance FROM data_view GROUP BY HiringDecision").toPandas()
strategy_count = spark.sql("SELECT RecruitmentStrategy, HiringDecision, COUNT(*) AS Count FROM data_view GROUP BY RecruitmentStrategy, HiringDecision").toPandas()
previous_companies_avg = spark.sql("SELECT HiringDecision, AVG(PreviousCompanies) AS AvgPreviousCompanies FROM data_view GROUP BY HiringDecision").toPandas()

gender_decision_count_sql = spark.sql("SELECT Gender, HiringDecision, COUNT(*) AS Count FROM data_view GROUP BY Gender, HiringDecision").toPandas()
gender_decision_count = gender_decision_count_sql.pivot(index='Gender', columns='HiringDecision', values='Count')

def add_annotations(ax, data, x, y, fmt='.2f'):
    for p in ax.patches:
        ax.annotate(format(p.get_height(), fmt),
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha = 'center', va = 'center',
                    xytext = (0, 9),
                    textcoords = 'offset points')

plt.figure(figsize=(10, 6))
ax = sns.barplot(x='HiringDecision', y='AvgInterviewScore', data=interview_score_avg, palette='viridis')
plt.title('Moyenne des scores d\'entretien par décision de recrutement')
plt.xlabel('Décision de recrutement (0 = Refusé, 1 = Accepté)')
plt.ylabel('Score d\'entretien moyen')
add_annotations(ax, interview_score_avg, 'HiringDecision', 'AvgInterviewScore')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
ax = sns.barplot(x='HiringDecision', y='AvgSkillScore', data=skill_score_avg, palette='viridis')
plt.title('Moyenne des scores de compétences par décision de recrutement')
plt.xlabel('Décision de recrutement (0 = Refusé, 1 = Accepté)')
plt.ylabel('Score de compétences moyen')
add_annotations(ax, skill_score_avg, 'HiringDecision', 'AvgSkillScore')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
ax = sns.barplot(x='HiringDecision', y='AvgPersonalityScore', data=personality_score_avg, palette='viridis')
plt.title('Moyenne des scores de personnalité par décision de recrutement')
plt.xlabel('Décision de recrutement (0 = Refusé, 1 = Accepté)')
plt.ylabel('Score de personnalité moyen')
add_annotations(ax, personality_score_avg, 'HiringDecision', 'AvgPersonalityScore')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
ax = sns.barplot(x='HiringDecision', y='AvgAge', data=age_avg, palette='viridis')
plt.title('Moyenne des âges par décision de recrutement')
plt.xlabel('Décision de recrutement (0 = Refusé, 1 = Accepté)')
plt.ylabel('Âge moyen')
add_annotations(ax, age_avg, 'HiringDecision', 'AvgAge')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
ax = sns.barplot(x='HiringDecision', y='AvgExperienceYears', data=experience_years_avg, palette='viridis')
plt.title('Moyenne des années d\'expérience par décision de recrutement')
plt.xlabel('Décision de recrutement (0 = Refusé, 1 = Accepté)')
plt.ylabel('Années d\'expérience moyennes')
add_annotations(ax, experience_years_avg, 'HiringDecision', 'AvgExperienceYears')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
ax = sns.barplot(x='EducationLevel', y='Count', hue='HiringDecision', data=education_count, palette='viridis')
plt.title('Nombre de candidats par niveau d\'éducation en fonction de la décision de recrutement')
plt.xlabel('Niveau d\'éducation (1 = Licence (Type 1), 2 = Licence (Type 2), 3 = Master, 4 = Doctorat)')
plt.ylabel('Nombre de candidats')
handles, labels = ax.get_legend_handles_labels()
labels = ['Refusé (0)', 'Accepté (1)']
ax.legend(handles=handles, title='Décision de recrutement', labels=labels)

add_annotations(ax, education_count, 'EducationLevel', 'Count')
plt.grid(True)
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(14, 7))
gender_decision_count[0].plot.pie(ax=axes[0], autopct='%1.1f%%', startangle=90, title='Répartition des refusés par genre')
gender_decision_count[1].plot.pie(ax=axes[1], autopct='%1.1f%%', startangle=90, title='Répartition des acceptés par genre')
axes[0].legend(labels=['Homme (0)', 'Femme (1)'], loc="best")
axes[1].legend(labels=['Homme (0)', 'Femme (1)'], loc="best")
plt.suptitle('Répartition des candidats par genre et décision de recrutement')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
ax = sns.barplot(x='RecruitmentStrategy', y='Count', hue='HiringDecision', data=strategy_count, palette='viridis')
plt.title('Nombre de candidats par stratégie de recrutement en fonction de la décision de recrutement')
plt.xlabel('Stratégie de recrutement (1 = Agressive, 2 = Modérée, 3 = Prudente)')
plt.ylabel('Nombre de candidats')
handles, labels = ax.get_legend_handles_labels()
labels = ['Refusé (0)', 'Accepté (1)']
ax.legend(handles=handles, title='Décision de recrutement', labels=labels)
add_annotations(ax, strategy_count, 'RecruitmentStrategy', 'Count')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
ax = sns.boxplot(x='HiringDecision', y='DistanceFromCompany', data=df.toPandas(), palette='viridis')
plt.title('Moyenne des distances par décision de recrutement')
plt.xlabel('Décision de recrutement (0 = Refusé, 1 = Accepté)')
plt.ylabel('Distance de la compagnie (km)')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
ax = sns.barplot(x='HiringDecision', y='AvgPreviousCompanies', data=previous_companies_avg, palette='viridis')
plt.title('Moyenne des compagnies précédentes par décision de recrutement')
plt.xlabel('Décision de recrutement (0 = Refusé, 1 = Accepté)')
plt.ylabel('Nombre moyen de compagnies précédentes')
add_annotations(ax, previous_companies_avg, 'HiringDecision', 'AvgPreviousCompanies')
plt.grid(True)
plt.show()

plt.figure(figsize=(14, 10))
correlation_matrix = df.toPandas().corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Heatmap de corrélation entre toutes les variables')
plt.show()

spark.stop()
