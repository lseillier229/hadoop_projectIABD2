from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, count, lit, round
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

spark = SparkSession.builder \
    .appName("PySpark Hadoop Example") \
    .config("spark.hadoop.fs.defaultFS", "hdfs://localhost:9000") \
    .getOrCreate()

df = spark.read.csv("hdfs://localhost:9000/data.csv", header=True, inferSchema=True)



print("Types de données avant le nettoyage:")
df.printSchema()

df.show(5)

print("Types de données après la transformation:")
df.printSchema()

print("Valeurs uniques pour RecruitmentStrategy:")
df.select('RecruitmentStrategy').distinct().show()

print("Valeurs uniques pour HiringDecision:")
df.select('HiringDecision').distinct().show()

print("Valeurs uniques pour Gender:")
df.select('Gender').distinct().show()

df_pd = df.toPandas()

def plot_pie_chart(data, column, ax):
    counts = data[column].value_counts()
    counts.plot.pie(ax=ax, autopct='%1.1f%%', startangle=90)
    ax.set_title(f'Distribution de {column}')
    ax.axis('equal')

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

plot_pie_chart(df_pd, 'RecruitmentStrategy', axes[0, 0])
plot_pie_chart(df_pd, 'HiringDecision', axes[0, 1])
plot_pie_chart(df_pd, 'Gender', axes[0, 2])
plot_pie_chart(df_pd, 'EducationLevel', axes[1, 0])
plot_pie_chart(df_pd, 'ExperienceYears', axes[1, 1])
plot_pie_chart(df_pd, 'PreviousCompanies', axes[1, 2])

plt.suptitle('Distribution des types de données par colonne')
plt.tight_layout()
plt.show()

interview_score_avg = df.groupBy('HiringDecision').agg(avg('InterviewScore').alias('AvgInterviewScore')).toPandas()

skill_score_avg = df.groupBy('HiringDecision').agg(avg('SkillScore').alias('AvgSkillScore')).toPandas()

personality_score_avg = df.groupBy('HiringDecision').agg(avg('PersonalityScore').alias('AvgPersonalityScore')).toPandas()

age_avg = df.groupBy('HiringDecision').agg(avg('Age').alias('AvgAge')).toPandas()

experience_years_avg = df.groupBy('HiringDecision').agg(avg('ExperienceYears').alias('AvgExperienceYears')).toPandas()

education_count = df.groupBy('EducationLevel', 'HiringDecision').agg(count('*').alias('Count')).toPandas()

gender_count = df.groupBy('Gender', 'HiringDecision').agg(count('*').alias('Count')).toPandas()

distance_avg = df.groupBy('HiringDecision').agg(round(avg('DistanceFromCompany'), 2).alias('AvgDistance')).toPandas()

strategy_count = df.groupBy('RecruitmentStrategy', 'HiringDecision').agg(count('*').alias('Count')).toPandas()

previous_companies_avg = df.groupBy('HiringDecision').agg(avg('PreviousCompanies').alias('AvgPreviousCompanies')).toPandas()

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
plt.title('Nombre de candidats par niveau d\'éducation et décision de recrutement')
plt.xlabel('Niveau d\'éducation')
plt.ylabel('Nombre de candidats')
plt.legend(title='Décision de recrutement', labels=['Refusé (0)', 'Accepté (1)'])
add_annotations(ax, education_count, 'EducationLevel', 'Count')
plt.grid(True)
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(14, 7))
gender_decision_count = df_pd.groupby(['Gender', 'HiringDecision']).size().unstack()
gender_decision_count[0].plot.pie(ax=axes[0], autopct='%1.1f%%', startangle=90, title='Répartition des refusés par genre')
gender_decision_count[1].plot.pie(ax=axes[1], autopct='%1.1f%%', startangle=90, title='Répartition des acceptés par genre')
axes[0].legend(labels=['Homme (0)', 'Femme (1)'], loc="best")
axes[1].legend(labels=['Homme (0)', 'Femme (1)'], loc="best")
plt.suptitle('Répartition des candidats par genre et décision de recrutement')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
ax = sns.barplot(x='RecruitmentStrategy', y='Count', hue='HiringDecision', data=strategy_count, palette='viridis')
plt.title('Nombre de candidats par stratégie de recrutement et décision de recrutement')
plt.xlabel('Stratégie de recrutement')
plt.ylabel('Nombre de candidats')
plt.legend(title='Décision de recrutement', labels=['Refusé (0)', 'Accepté (1)'])
add_annotations(ax, strategy_count, 'RecruitmentStrategy', 'Count')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
ax = sns.boxplot(x='HiringDecision', y='DistanceFromCompany', data=df_pd, palette='viridis')
plt.title('Moyenne des distances par décision de recrutement')
plt.xlabel('Décision de recrutement (0 = Refusé, 1 = Accepté)')
plt.ylabel('Distance de la compagnie (m)')
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


plt.figure(figsize=(14, 11))
correlation_matrix = df_pd.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Heatmap de corrélation entre toutes les variables')
plt.show()

spark.stop()
