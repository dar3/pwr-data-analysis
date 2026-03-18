import numpy as np
from scipy import stats

# Exercise 1

np.random.seed(42) # for same results
travelling = 5 * np.random.randn(100) + 31.5

#t - test
# H0 mean = 28
t_stat, p_val = stats.ttest_1samp(travelling, popmean=28)

print(f"Statistics t: {t_stat:.4f}")
print(f"Value p: {p_val:.4e}")

if p_val < 0.05:
    print("Reject H0: The average travel time is different from 28 min.")
else:
    print("Don't reject H0.")


# Exercise 2
print()
print("Exercise 2")
delivery_time = [1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 6, 6, 6, 7, 7]

# H0: m = 3
# popmean - wartosc oczekiwana
t_stat, p_val = stats.ttest_1samp(delivery_time, popmean=3)

print(f"Mean: {np.mean(delivery_time):.2f}")
print(f"p value: {p_val:.4f}")

if p_val < 0.05:
    print("Reject H0")
else:
    print("Do not reject H0")
# if p-value is small (< 0.05), reject hypothesis about 3 days

# Exercise 3
print()
print("Exercise 3")
technical_norm_time = 6.0              # m0 H0
employees = 25
mean_of_try = 6 + 20 / 60
sigma = 1.5

# Z = (meann - technical_norm_time) / (sigma / sqrt(employees))
z_stat = (mean_of_try - technical_norm_time) / (sigma / np.sqrt(employees))

# Calculation of p value
# Check where it takes more time than technical_norm (one side test)
p_val = 1 - stats.norm.cdf(z_stat)

print(f"Statistic Z: {z_stat:.4f}")
print(f"Value p: {p_val:.4f}")

alpha = 0.05
if p_val <= alpha:
    print("reject H0 and choose H1: technical time is too short")
else:
    print("No basis to reject H0")

# Exercise 4
print(f"\nExercise 4")
n = 25
mean_time = 6 + 20 / 60
sigma_h0 = 1.5
sigma_check_value = 1.6

np.random.seed(42)
proba = np.random.normal(mean_time, sigma_h0, n)


# same as vartest
def var_test(dane, sigma_h0, alpha, alternative='smaller'):
    n = len(dane)
    s2 = np.var(dane, ddof=1)
    # Chi kwadrat
    chi2_stat = (n - 1) * s2 / (sigma_h0 ** 2)

    # Counting p value one side test
    p_val = stats.chi2.cdf(chi2_stat, n - 1)

    return chi2_stat, p_val


#  a = 0.05
stat5, p5 = var_test(proba, sigma_check_value, 0.05)
print(f" alpha = 0.05: p-value = {p5:.4f}")
print("reject H0" if p5 < 0.05 else "No reasons to reject H0")


stat10, p10 = var_test(proba, sigma_check_value, 0.1)
print(f"\n alpha = 0.1: p-value = {p10:.4f}")
print("reject H0" if p10 < 0.1 else "No reasons to reject H0")

# Exercise 5

print(f"\nExcercise 5")
n1, mean1, std1 = 20, 27.7, 5.5   # New product buyers
n2, mean2, std2 = 22, 32.1, 6.3   # Known product buyers

np.random.seed(42)
new_product_group = np.random.normal(mean1, std1, n1)
known_product_group = np.random.normal(mean2, std2, n2)

# F test - same as vartest2
# Calculate variances from the generated samples.
s1_sq = np.var(new_product_group, ddof=1)
s2_sq = np.var(known_product_group, ddof=1)

# Statistic F = s1^2 / s2^2
f_stat = s1_sq / s2_sq
# df (freedom degree)
df1 = n1 - 1
df2 = n2 - 1

# P-value for one side test
p_val = 2 * min(stats.f.cdf(f_stat, df1, df2), 1 - stats.f.cdf(f_stat, df1, df2))

print(f"Group variance 1: {s1_sq:.2f} (std dev: {np.sqrt(s1_sq):.2f})")

print(f"Group variance 2: {s2_sq:.2f} (std dev: {np.sqrt(s2_sq):.2f})")
print(f"F-statistic: {f_stat:.4f}")
print(f"p-value: {p_val:.4f}")

alpha = 0.05
if p_val <= alpha:
    print("Reject H0: difference between standard deviations is significant (statistically).")
else:
    print("Cannot reject H0: There is no big difference between the standard deviations.")