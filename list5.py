import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import pandas as pd
import pandas as pd
import numpy as np
from scipy.stats import binomtest

data_czyt = pd.read_csv('data/czytelnictwo.csv', sep=None, engine='python')
data_chmiel = pd.read_csv('data/chmiel.csv', sep=None, engine='python')
data_koronografia = pd.read_csv('data/dane z koronografii.csv', sep=None, engine='python')

# Ex. 1
w1 = np.array([88, 69, 86, 59, 57, 82, 94, 93, 64, 91, 86, 59, 91, 60, 57, 92, 70, 88, 70, 85])
w2 = np.array([73, 68, 75, 54, 53, 84, 84, 86, 66, 84, 78, 58, 91, 57, 59, 88, 71, 84, 64, 85])

plt.figure(figsize=(8, 6))
plt.boxplot([w1, w2], tick_labels=['Przed dietą (w1)', 'Po diecie (w2)'])
plt.title('Rozkład ciezaru ciała przed i po 7 tygodniowej diecie')
plt.ylabel('Ciężar ciała (kg)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

plt.figure(figsize=(8, 6))
for i in range(len(w1)):
    plt.plot([1, 2], [w1[i], w2[i]], marker='o', color='blue', alpha=0.5)
plt.xticks([1, 2], ['Przed dietą', 'Po diecie'])
plt.title('Zmiany ciężaru ciała dla poszczególnych kobiet')
plt.ylabel('Ciężar ciała (kg)')
plt.xlim(0.5, 2.5)
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# test for pairs alternative='greater', because we expect, that w1 > w2 (smaller weight)
stat, p_value = stats.wilcoxon(w1, w2, alternative='greater')

print("Ex. 1")
print("Wyniki testu kolejności par Wilcoxona")
print(f"Statystyka testowa (W): {stat}")
print(f"Wartość p-value: {p_value:.5f}")

alpha = 0.05
if p_value < alpha:
    print("\nWniosek: Odrzucamy hipotezę zerową.")
    print("Otrzymane wyniki przeczą hipotezie, że dieta nie powoduje zmniejszenia ciężaru ciała.")
else:
    print("\nWniosek: Brak podstaw do odrzucenia hipotezy zerowej.")
    print("Nie można stwierdzić, że dieta statystycznie istotnie zmniejsza ciężar ciała.")


# Ex. 2
df_czytelnictwo = pd.read_csv('data/czytelnictwo.csv', sep=None, engine='python')
print("Ex. 2")
print("Test Znaków")

time_before = df_czytelnictwo.iloc[:, 0].dropna()
time_after = df_czytelnictwo.iloc[:, 1].dropna()

roznice_czyt = time_before - time_after

roznice_czyt_bez_zer = roznice_czyt[roznice_czyt != 0]

zmiany_kierunkowe = np.sum(roznice_czyt_bez_zer > 0)
n_czyt = len(roznice_czyt_bez_zer)

wynik_czyt = binomtest(zmiany_kierunkowe, n_czyt, p=0.5, alternative='two-sided')

print(f"Liczba badanych par po odrzuceniu zer: {n_czyt}")
print(f"P-value: {wynik_czyt.pvalue:.5f}")

poziom_istotnosci = 0.05
if wynik_czyt.pvalue < poziom_istotnosci:
    print("\nWniosek: Odrzucamy hipotezę zerową.")
    print("Zatrudnienie w firmie miało statystycznie istotny wpływ na czas poświęcany na lekturę prasy.")
else:
    print("\nWniosek: Brak podstaw do odrzucenia hipotezy zerowej.")
    print("Brak dowodów statystycznych na to, że zatrudnienie wpłynęło na czas czytania.")


# Ex. 3
print("\nEx. 3 Chmiel")
pollinated = data_chmiel.iloc[:, 0].dropna()
not_pollinated = data_chmiel.iloc[:, 1].dropna()


stat_chmiel, p_val_chmiel = stats.wilcoxon(pollinated, not_pollinated, alternative='two-sided')

print(f"Statystyka W: {stat_chmiel}")
print(f"P-value: {p_val_chmiel:.5f}")

alpha_chmiel = 0.05
if p_val_chmiel < alpha_chmiel:
    print("Wniosek:")
    print("\nOdrzucamy hipotezę zerową (H0).")
    print("Zapylenie ma istotny wpływ na masę nasion.")
else:
    print("Wniosek:")
    print("\nBrak podstaw do odrzucenia hipotezy zerowej (H0).")
    print("Brak dowodów na to, by zapylenie miało wpływ na masę nasion.")


# Ex. 4
before_empl = data_czyt.iloc[:, 0].dropna()
after_empl = data_czyt.iloc[:, 1].dropna()

# Weryfikacja zmiany czasu (test Wilcoxona dla par)
statystyka_w, p_val = stats.wilcoxon(before_empl, after_empl, alternative='two-sided')

print("\nEx. 4")
print(f"Wartość statystyki Wilcoxona: {statystyka_w}")
print(f"Wyznaczone p-value: {p_val:.5f}")

alfa = 0.05
if p_val < alfa:
    print("\nWniosek:")
    print("Czas na czytanie prasy uległ istotnej zmianie po przyjęciu osób do pracy")
else:
    print("\nWniosek:")
    print("Nie stwierdzono statystycznie istotnej zmiany w ilości czasu poświęcanego na czytanie prasy po otrzymaniu pracy.")

# Ex. 5
print("\nEx. 5 Koronografia")
time_gr1 = data_koronografia[data_koronografia['group'] == 1]['time'].dropna()
time_gr2 = data_koronografia[data_koronografia['group'] == 2]['time'].dropna()

stat_korono, p_val_korono = stats.mannwhitneyu(time_gr1, time_gr2, alternative='two-sided')

print(f"Statystyka U: {stat_korono}")
print(f"P-value: {p_val_korono:.5f}")

# alpha_korono = 1.0 - poziom istotnosci (1.0 - 0.9 = 0.1)
alpha_korono = 0.1
if p_val_korono < alpha_korono:
    print("\nWniosek: Odrzucamy hipotezę zerową (H0).")
    print("Czas ćwiczenia zależy od stanu zdrowia (różnice między grupami są istotne statystycznie).")
else:
    print("\nWniosek: Brak podstaw do odrzucenia hipotezy zerowej (H0).")
    print("Nie można stwierdzić przy poz. ufn. 0.9, by czas ćwiczenia zależał od stanu zdrowia.")