import pandas as pd
import numpy as np
from scipy import stats
import scipy.io as sio
import matplotlib.pyplot as plt

def check_normality_and_plot(data, name):
    stat, p = stats.shapiro(data)
    print(f"[{name}] Test Shapiro-Wilka: stat={stat:.4f}, p-value={p:.4f}")
    if p > 0.05:
        print(" -> Brak podstaw do odrzucenia H0. Zakladamy rozklad normalny.")
    else:
        print(" -> Odrzucamy H0. Rozklad rozni sie od normalnego (uwaga dla testu t!).")

    plt.figure(figsize=(5, 4))
    stats.probplot(data, dist="norm", plot=plt)
    plt.title(f"Wykres Q-Q: {name}")
    plt.grid(True)
    plt.show()

def check_variance(data1, data2, name1, name2):

    stat, p = stats.levene(data1, data2)
    print(f"[Wariancja: {name1} vs {name2}] Test Levene'a: stat={stat:.4f}, p-value={p:.4f}")
    if p > 0.05:
        print(" -> Brak podstaw do odrzucenia H0. Zakladamy rownosc wariancji (equal_var=True).")
        return True
    else:
        print(" -> Odrzucamy H0. Wariancje sa rozne (wykorzystamy test Welcha: equal_var=False).")
        return False

nervous = np.array([3, 3, 4, 5, 5])
calm = np.array([4, 6, 7, 9, 9])
before30 = np.array([6, 7, 10, 9])
after30 = np.array([5, 6, 2, 3])
data13 = np.array([175.26, 177.8, 167.64, 160.02, 172.72, 177.8, 175.26, 170.18, 157.48, 160.02, 193.04, 149.86, 157.48, 157.48, 190.5, 157.48, 182.88, 160.02])
data17 = np.array([172.72, 157.48, 170.18, 172.72, 175.26, 170.18, 154.94, 149.86, 157.48, 154.94, 175.26, 167.64, 157.48, 157.48, 154.94, 177.8])


print("\n--- Ex. 1 ---")
men_mat = sio.loadmat('data/body_men.mat')
women_mat = sio.loadmat('data/body_women.mat')
men_feature = men_mat['body_men'][:, 1]
women_feature = women_mat['body_women'][:, 1]

check_normality_and_plot(men_feature, "Mezczyzni")
check_normality_and_plot(women_feature, "Kobiety")

# Sprawdzenie wariancji przed testem
is_var_equal = check_variance(men_feature, women_feature, "Mezczyzni", "Kobiety")

print("\nWyniki wg polecenia (wymuszenie parametrow):")
t_stat_eq, p_val_eq = stats.ttest_ind(men_feature, women_feature, equal_var=True, alternative='two-sided')
print(f"T-test (equal_var=True): t={t_stat_eq:.4f}, p={p_val_eq:.4f}")
t_stat_uneq, p_val_uneq = stats.ttest_ind(men_feature, women_feature, equal_var=False, alternative='two-sided')
print(f"T-test Welcha (equal_var=False): t={t_stat_uneq:.4f}, p={p_val_uneq:.4f}")


print("\n--- Ex. 2 ---")
print("H0: Srednia gestykulacja u nerwowych = u spokojnych (u1 = u2)")
print("H1: Osoby nerwowe gestykuluja wiecej (u1 > u2) test jednostronny")

check_normality_and_plot(nervous, "Nerwowi")
check_normality_and_plot(calm, "Spokojni")

is_var_equal_ex2 = check_variance(nervous, calm, "Nerwowi", "Spokojni")

t_stat2, p_val2 = stats.ttest_ind(nervous, calm, equal_var=is_var_equal_ex2, alternative='greater')
print(f"t-stat = {t_stat2:.4f}, p-value = {p_val2:.4f}")
if p_val2 < 0.05:
    print("Odrzucamy H0. Nerwowi gestykuluja wiecej.")
else:
    print("Brak podstaw do odrzucenia H0. Nerwowi NIE gestykuluja wiecej.")


print("\n--- Ex. 3 ---")
print("H0: Obie grupy sa tak samo dowcipne (u1 = u2)")
print("H1: Osoby < 30 sa bardziej dowcipne (u1 > u2) - test jednostronny")

check_normality_and_plot(before30, "Przed 30")
check_normality_and_plot(after30, "Po 30")

is_var_equal_ex3 = check_variance(before30, after30, "Przed 30", "Po 30")

t_stat3, p_val3 = stats.ttest_ind(before30, after30, equal_var=is_var_equal_ex3, alternative='greater')
print(f"t-stat = {t_stat3:.4f}, p-value = {p_val3:.4f}")
if p_val3 < 0.05:
    print("Odrzucamy H0. Osoby <30 sa bardziej dowcipne.")
else:
    print("Brak podstaw do odrzucenia H0.")



print("\n--- Ex. 4 ---")
print("Nie mozna wykonac testu dla prob zaleznych, poniewaz grupy mezczyzn i kobiet "
      "sa niezalezne, dotycza innych osob i maja rozna liczebnosc. Test zalezny "
      "wymaga par. Nalezy stworzyc sztuczne pary (matching).")



print("\n--- Ex. 5 ---")
check_normality_and_plot(data13, "Wzrost - Grupa 13:00")

# W teście dla 1 próby nie sprawdzamy wariancji (nie ma z czym jej porównać).
t_stat5, p_val5 = stats.ttest_1samp(data13, 169.051)
print(f"p-value = {p_val5:.4f}")
if p_val5 < 0.05:
    print("Odrzucamy H0. Srednia rozni sie od 169.051.")
else:
    print("Brak podstaw do odrzucenia H0.")



print("\n--- Ex. 6 ---")
check_normality_and_plot(data17, "Wzrost - Grupa 17:00")

t_stat6, p_val6 = stats.ttest_1samp(data17, 164.1475)
print(f"p-value = {p_val6:.4f}")
if p_val6 < 0.05:
    print("Odrzucamy H0. Srednia rozni sie od 164.1475.")
else:
    print("Brak podstaw do odrzucenia H0.")



print("\n--- Ex. 7 ---")
print("H0: Rozklady obu grup sa rowne")
print("H1: Osoby nerwowe gestykuluja silniej -> jednostronny")
# Test U nie wymaga normalności ani równej wariancji, więc nie sprawdzamy ich tutaj.

u_stat7, p_val7 = stats.mannwhitneyu(nervous, calm, alternative='greater')
print(f"U-stat = {u_stat7:.1f}, p-value = {p_val7:.4f}")
if p_val7 < 0.05:
    print("Odrzucamy H0.")
else:
    print("Brak podstaw do odrzucenia H0.")



print("\n--- Ex. 8 ---")
print("H0: Wzrost studentow w obu grupach jest rownie duzy.")
print("H1: Wzrost w obu grupach sie rozni (test dwustronny).")

u_stat8, p_val8 = stats.mannwhitneyu(data13, data17, alternative='two-sided')
print(f"U-stat = {u_stat8:.1f}, p-value = {p_val8:.4f}")
if p_val8 < 0.05:
    print("Odrzucamy H0, wzrost w grupach sie rozni.")
else:
    print("Brak podstaw do odrzucenia H0.")