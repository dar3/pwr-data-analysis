import pandas as pd
import numpy as np
from scipy import stats
import scipy.io as sio
import matplotlib.pyplot as plt


def check_normality_and_plot(data, name):
    stat, p = stats.shapiro(data)
    print(f"[{name}] Test Shapiro-Wilka: stat={stat:.4f}, p-value={p:.4f}")
    if p > 0.05:
        print("No reasons to reject H0. Assume normal distribution.")
    else:
        print("Reject H0. Distribution differs from the normal one (be cautious for t-test).")

    plt.figure(figsize=(5, 4))
    stats.probplot(data, dist="norm", plot=plt)
    plt.title(f"Wykres Q-Q: {name}")
    plt.grid(True)
    plt.show()



nervous = np.array([3, 3, 4, 5, 5])
calm = np.array([4, 6, 7, 9, 9])
before30 = np.array([6, 7, 10, 9])
after30 = np.array([5, 6, 2, 3])
data13 = np.array(
    [175.26, 177.8, 167.64, 160.02, 172.72, 177.8, 175.26, 170.18, 157.48, 160.02, 193.04, 149.86, 157.48, 157.48,
     190.5, 157.48, 182.88, 160.02])
data17 = np.array(
    [172.72, 157.48, 170.18, 172.72, 175.26, 170.18, 154.94, 149.86, 157.48, 154.94, 175.26, 167.64, 157.48, 157.48,
     154.94, 177.8])


# Ex. 1 Comparing temp for women and men
print("\n Ex. 1")
men_mat = sio.loadmat('data/body_men.mat')
women_mat = sio.loadmat('data/body_women.mat')
men_feature = men_mat['body_men'][:, 1]
women_feature = women_mat['body_women'][:, 1]

# Checking normal distribution
check_normality_and_plot(men_feature, "Mezczyzni (Cechy)")
check_normality_and_plot(women_feature, "Kobiety (Cechy)")

t_stat_eq, p_val_eq = stats.ttest_ind(men_feature, women_feature, equal_var=True, alternative='two-sided')
print(f"T-test (equal_var=True): t={t_stat_eq:.4f}, p={p_val_eq:.4f}")
t_stat_uneq, p_val_uneq = stats.ttest_ind(men_feature, women_feature, equal_var=False, alternative='two-sided')
print(f"T-test Welcha (equal_var=False): t={t_stat_uneq:.4f}, p={p_val_uneq:.4f}")


# Ćwiczenie 2: Nervous vs Calm
print("\n Ex. 2")
print("H0: Srednia gestykulacja u nerwowych = u spokojnych (u1 = u2)")
print("H1: Osoby nerwowe gestykuluja więcej (u1 > u2) test jednostronny (prawostronny / greater)")
df_ex2 = len(nervous) + len(calm) - 2
print(f"Stopnie swobody (df): {len(nervous)} + {len(calm)} - 2 = {df_ex2}")

check_normality_and_plot(nervous, "Gestykulacja - Nerwowi")
check_normality_and_plot(calm, "Gestykulacja - Spokojni")

t_stat2, p_val2 = stats.ttest_ind(nervous, calm, alternative='greater')
print(f"t-stat = {t_stat2:.4f}, p-value = {p_val2:.4f}")
if p_val2 < 0.05:
    print("Odrzucamy H0. Nerwowi gestykuluja więcej.")
else:
    print("Brak podstaw do odrzucenia H0. W naszej probie nerwowi NIE gestykuluja wiecej.")


# Ex. 3: Below 30 vs Above 30 (Rozbawienie)
print("\n Ex. 3")
print("H0: Obie grupy są tak samo dowcipne (u1 = u2)")
print("H1: Osoby < 30 są bardziej dowcipne (u1 > u2) - test jednostronny (prawostronny / greater)")
df_ex3 = len(before30) + len(after30) - 2
print(f"Stopnie swobody (df): {len(before30)} + {len(after30)} - 2 = {df_ex3}")

check_normality_and_plot(before30, "Dowcip - Poniżej 30")
check_normality_and_plot(after30, "Dowcip - Po 30")

t_stat3, p_val3 = stats.ttest_ind(before30, after30, alternative='greater')
print(f"t-stat = {t_stat3:.4f}, p-value = {p_val3:.4f}")
if p_val3 < 0.05:
    print("Odrzucamy H0. Osoby <30 są bardziej dowcipne.")
else:
    print("Brak podstaw do odrzucenia H0.")


# Ex. 4: Graduates - Test prob zaleznych
print("\n Ex. 4")
print("Nie mozna wykonac testu dla prob zaleznych, poniewaz grupy mezczyzn i kobiet "
      "sa niezalezne, dotycza innych osob i maja rozna liczebność. Test zalezny "
      "wymaga par (np. pomiar tej samej grupy osób przed i po). "
      "Aby uzyć testu t-zaleznego, trzeba stworzyc sztuczne pary (matching).")


# Ex. 5: Avg height for group from 13:00 = 169.051
print("\n Ex. 5")
check_normality_and_plot(data13, "Wzrost - Grupa 13:00")
df_ex5 = len(data13) - 1
print(f"Stopnie swobody (df) dla jednej proby: n - 1 = {df_ex5}")

t_stat5, p_val5 = stats.ttest_1samp(data13, 169.051)
print(f"p-value = {p_val5:.4f}")
if p_val5 < 0.05:
    print("Odrzucamy H0. Średnia różni się od 169.051.")
else:
    print("Brak podstaw do odrzucenia H0. Srednia statystycznie wynosi 169.051.")


# Ex. 6: Avg height for group from 17:00 = 164.1475
print("\nEx. 6")
check_normality_and_plot(data17, "Wzrost - Grupa 17:00")
df_ex6 = len(data17) - 1
print(f"Stopnie swobody (df): n - 1 = {df_ex6}")

t_stat6, p_val6 = stats.ttest_1samp(data17, 164.1475)
print(f"p-value = {p_val6:.4f}")
if p_val6 < 0.05:
    print("Odrzucamy H0. Srednia rozni się od 164.1475.")
else:
    print("Brak podstaw do odrzucenia H0. Średnia statystycznie wynosi 164.1475.")


# Ex. 7: Nervous vs calm - Test U Manna-Whitneya
print("\nEx. 7")
print("H0: Rozkłady obu grup są równe (P(X>Y) = P(Y>X))")
print("H1: Osoby nerwowe gestykulują silniej -> jednostronny")
print("Mozna zastosowac test U. Jest to alternatywa, gdy próbka jest mała (po 5 obserwacji) "
      "a zmienna (liczba gestów) ma charakter porządkowy/ilościowy.")

u_stat7, p_val7 = stats.mannwhitneyu(nervous, calm, alternative='greater')
print(f"U-stat = {u_stat7:.1f}, p-value = {p_val7:.4f}")
if p_val7 < 0.05:
    print("Odrzucamy H0.")
else:
    print("Brak podstaw do odrzucenia H0.")


# Ex. 8 Students height, gr 13 vs gr 17 - Test U
print("\nEx. 8")
print("H0: Wzrost studentów w obu grupach jest równie duży.")
print("H1: Wzrost w obu grupach się różni (test dwustronny).")
u_stat8, p_val8 = stats.mannwhitneyu(data13, data17, alternative='two-sided')
print(f"U-stat = {u_stat8:.1f}, p-value = {p_val8:.4f}")
if p_val8 < 0.05:
    print("Odrzucamy H0, wzrost w grupach się różni.")
else:
    print("Brak podstaw do odrzucenia H0, wzrost w grupach nie wykazuje istotnych różnic.")