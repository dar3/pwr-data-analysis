import pandas as pd
import numpy as np
from scipy import stats

nervous = np.array([3, 3, 4, 5, 5])
calm = np.array([4, 6, 7, 9, 9])
before30 = np.array([6, 7, 10, 9])
after30 = np.array([5, 6, 2, 3])

data13 = np.array([175.26, 177.8, 167.64, 160.02, 172.72, 177.8, 175.26, 170.18, 157.48, 160.02, 193.04, 149.86, 157.48, 157.48, 190.5, 157.48, 182.88, 160.02])

data17 = np.array([172.72, 157.48, 170.18, 172.72, 175.26, 170.18, 154.94, 149.86, 157.48, 154.94, 175.26, 167.64, 157.48, 157.48, 154.94, 177.8])

# Exercise 1
# stats.ttest_ind(men, women, equal_var=True/False, alternative='two-sided'/'less'/'greater')

# compare alpha with p manually

#Test t dla prób niezależnych
# H0: u1 = u2 (no difference), H1: u1 > u2 (nervous gesticulate more)

t_stat2, p_val2 = stats.ttest_ind(nervous, calm, alternative='greater')
print("\nEx. 2")
print(f"t-stat = {t_stat2:.4f}, p-value = {p_val2:.4f}")
if p_val2 < 0.05:
    print("Reject H0 and choose H1")
else:
    print("No reason to reject H0")

# Interpretation: p > 0.05, no reasons to reject H0 (here nervous gesticulated less).

# Test t dla prób niezależnych
# H0: u1 = u2, H1: u1 > u2 (before 30 are more humorous)
# one side - right - greater
t_stat3, p_val3 = stats.ttest_ind(before30, after30, alternative='greater')
print("\nEx. 3")
print(f"t-stat = {t_stat3:.4f}, p-value = {p_val3:.4f}")
if p_val3 < 0.05:
    print("Reject H0 and choose H1")
else:
    print("No reason to reject H0")


#Exercise 4: Graduates
abs_df = pd.read_csv('data/absolwenci.csv', sep=';', encoding='cp1250')
men_sal = abs_df[abs_df['GENDER'] == 'Mezczyzna']['SALARY']
women_sal = abs_df[abs_df['GENDER'] == 'Kobieta']['SALARY']

#Test dla prób zależnych (dependend?) requires to have pairs
#  631 mens and 469 women - the groups are unequal in size and independent.
# Cannot make test t-zależny. Solution: Create pairs (matching).

# Checking if students average height from group 13:00 is 169.051

t_stat5, p_val5 = stats.ttest_1samp(data13, 169.051)
print("\nEx. 5")
print(f"p-value = {p_val5:.4f}")
if p_val5 > 0.05:
    print("No reason to reject H0 (Avg is 169.051)")
else:
    print("Reject H0 choose H1")


# Checking if students average height from group 17:00 is 164.1475
t_stat6, p_val6 = stats.ttest_1samp(data17, 164.1475)
print("\nEx. 6")
print(f"p-value = {p_val6:.4f}")
if p_val6 > 0.05:
    print("No reason to reject H0 (Avg is 164.1475)")
else:
    print("Reject H0 choose H1")

# Mann-Whitney U test (Nieparametryczny)
# H0 nervous gesticulate as hard as calm
# H1 nervous gesticulate more than calm
u_stat7, p_val7 = stats.mannwhitneyu(nervous, calm, alternative='greater')
print("\nEx. 7")
print(f"U-stat = {u_stat7:.1f}, p-value = {p_val7:.4f}")
if p_val7 < 0.05:
    print("Reject H0 and choose H1")
else:
    print("No reasons to reject H0")

# two sided test
# H0 students height in both groups are the same
# H1 students height is different in groups
u_stat8, p_val8 = stats.mannwhitneyu(data13, data17)
print("\nEx. 8")
print(f"U-stat = {u_stat8:.1f}, p-value = {p_val8:.4f}")
if p_val8 < 0.05:
    print("Reject H0 and choose H1")
else:
    print("No reason to reject H0")