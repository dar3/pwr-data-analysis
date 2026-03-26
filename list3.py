import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.stats.diagnostic import lilliefors

# loading data
patients = pd.read_csv('data/pacjenci.csv', encoding='cp1250')
lightBulbs = pd.read_csv('data/zarowki.csv')
capacitors = pd.read_csv('data/kondensatory.csv')
graduates = pd.read_csv('data/absolwenci.csv', sep=';', encoding='cp1250')


controlB = np.array([0.08, 0.10, 0.15, 0.17, 0.24, 0.34, 0.38, 0.42, 0.49, 0.50, 0.70, 0.94, 0.95, 1.26, 1.37, 1.55, 1.75, 3.20, 6.98, 50.57])


def plot_creator(data, use_log=False):
    x = np.sort(data)
    y = np.arange(1, len(data) + 1) / len(data)

    plt.step(x, y, where='post')
    if use_log:
        plt.xscale('log')
        plt.title("ECDF - Logarythmic scale")
    else:
        plt.title("ECDF - Linear scale")
    plt.xlabel("Values")
    plt.ylabel("F(x)")
    plt.grid(True)
    plt.show()


# Ex 1
plot_creator(controlB, use_log=False)
# Ex 2
plot_creator(controlB, use_log=True)

# Ex 3
controlA = np.array([0.22, -0.87, -2.39, -1.79, 0.37, -1.54, 1.28, -0.31, -0.74, 1.72,
                     0.38, -0.17, -0.62, -1.10, 0.30, 0.15, 2.30, 0.19, -0.50, -0.09])
treatmentA = np.array([-5.13, -2.19, -2.43, -3.83, 0.50, -3.25, 4.32, 1.63, 5.18, -0.43,
                       7.11, 4.87, -3.10, -5.81, 3.76, 6.31, 2.58, 0.07, 5.76, 3.50])

# Kolmogorov-Smirnov test to check if distribution  of controlA and treatmentA are different
stat, p_val = stats.ks_2samp(controlA, treatmentA)
print("Exercise 3")
print(f"Test K-S: statistic={stat:.4f}, p-value={p_val:.4f}")

def get_ecdf(data):
    x = np.sort(data)
    y = np.arange(1, len(data) + 1) / len(data)
    return x, y

x1, y1 = get_ecdf(controlA)
x2, y2 = get_ecdf(treatmentA)


all_x = np.sort(np.concatenate([x1, x2]))
y1_interp = np.searchsorted(x1, all_x, side='right') / len(x1)
y2_interp = np.searchsorted(x2, all_x, side='right') / len(x2)
diffs = np.abs(y1_interp - y2_interp)
max_idx = np.argmax(diffs)
x_d = all_x[max_idx]
y_low = min(y1_interp[max_idx], y2_interp[max_idx])
y_high = max(y1_interp[max_idx], y2_interp[max_idx])


plt.figure(figsize=(8, 6))


plt.step(x1, y1, where='post', color='black', label='controlA', linewidth=1)
plt.step(x2, y2, where='post', color='black', linestyle='--', label='treatmentA', linewidth=1)

plt.vlines(x_d, y_low, y_high, color='red', linewidth=1.5)
plt.text(x_d + 0.2, (y_low + y_high) / 2, 'D', color='red', fontweight='bold', fontsize=14, fontstyle='italic')

plt.title("KS-Test Comparison Cumulative Fraction Plot")
plt.xlabel("X")
plt.ylabel("Fraction of total")
plt.ylim(0, 1.05)
plt.xlim(-6.5, 7.5)
plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
plt.grid(False)

plt.tick_params(direction='in', top=True, right=True, length=6)

plt.show()


# Exercise 4 and 6
men_h = patients[patients['plec'] == 'M']['wzrost']
women_h = patients[patients['plec'] == 'K']['wzrost']

print("Exercise 4")
stat4, p4 = stats.ks_2samp(men_h, women_h)
print(f"Test K-S (M vs K): p-value = {p4:.4f}")


delikates = np.array([23.4, 30.9, 18.8, 23.0, 21.4, 1, 24.6, 23.8, 24.1, 18.7, 16.3, 20.3, 14.9, 35.4, 21.6, 21.2, 21.0, 15.0, 15.6, 24.0, 34.6, 40.9, 30.7, 24.5, 16.6, 1, 21.7, 1, 23.6, 1, 25.7, 19.3, 46.9, 23.3, 21.8, 33.3, 24.9, 24.4, 1, 19.8, 17.2, 21.5, 25.5, 23.3, 18.6, 22.0, 29.8, 33.3, 1, 21.3, 18.6, 26.8, 19.4, 21.1, 21.2, 20.5, 19.8, 26.3, 39.3, 21.4, 22.6, 1, 35.3, 7.0, 19.3, 21.3, 10.1, 20.2, 1, 36.2, 16.7, 21.1, 39.1, 19.9, 32.1, 23.1, 21.8, 30.4, 19.62, 15.5])
renety = np.array([16.5, 1, 22.6, 25.3, 23.7, 1, 23.3, 23.9, 16.2, 23.0, 21.6, 10.8, 12.2, 23.6, 10.1, 24.4, 16.4, 11.7, 17.7, 34.3, 24.3, 18.7, 27.5, 25.8, 22.5, 14.2, 21.7, 1, 31.2, 13.8, 29.7, 23.1, 26.1, 25.1, 23.4, 21.7, 24.4, 13.2, 22.1, 26.7, 22.7, 1, 18.2, 28.7, 29.1, 27.4, 22.3, 13.2, 22.5, 25.0, 1, 6.6, 23.7, 23.5, 17.3, 24.6, 27.8, 29.7, 25.3, 19.9, 18.2, 26.2, 20.4, 23.3, 26.7, 26.0, 1, 25.1, 33.1, 35.0, 25.3, 23.6, 23.2, 20.2, 24.7, 22.6, 39.1, 26.5, 22.7])

# t-test assume normal distribution
t_stat, t_p = stats.ttest_ind(delikates, renety)
# Test K-S comparing distribution
ks_stat, ks_p = stats.ks_2samp(delikates, renety)

print("Exercise 5")
print(f"T-test p-value: {t_p:.4f}")
print(f"K-S test p-value: {ks_p:.4f}")


print("Exercise 6")
_, p6m = stats.shapiro(men_h)
_, p6w = stats.shapiro(women_h)
print(f"Normal distr (S-W) Mężczyźni: p = {p6m:.4f}")
print(f"Normal distr (S-W) Kobiety: p = {p6w:.4f}")
#
# stat_m_lilli, p6m_lilli = lilliefors(men_h, dist='norm')
# stat_w_lilli, p6w_lilli = lilliefors(women_h, dist='norm')
#
# print(f"Normalność Shapiro-Wilk (Mężczyźni): p = {p6m_sw:.4f}")
# print(f"Normalność Shapiro-Wilk (Kobiety): p = {p6w_sw:.4f}")
# print("-" * 30)
# print(f"Normalność Lilliefors (Mężczyźni): p = {p6m_lilli:.4f}")
# print(f"Normalność Lilliefors (Kobiety): p = {p6w_lilli:.4f}")


if p6m_lilli > 0.05 and p6w_lilli > 0.05:
    print("\nOba testy (S-W i Lilliefors) potwierdzają rozkład normalny wzrostu (p > 0.05).")


print("\n Exercise 7")
_, p7 = stats.shapiro(patients['cukier'])
print(f"Sugar norm: p = {p7:.4f}")

print("\n Exercise 8")
_, p8 = stats.shapiro(lightBulbs['czas'])
print(f"Light bulbs norm: p = {p8:.4f} (Alfa=0.1)")

print("\n Exercise 9")
_, p9 = stats.shapiro(capacitors['pojemnosc'])
print(f"Capacitors norm: p = {p9:.4f}")

print("\n Exercise 10 graduates")
rolnictwo_sal = graduates[graduates['COLLEGE'] == 'Rolnictwo']['SALARY']
pedagogika_sal = graduates[graduates['COLLEGE'] == 'Pedagogika']['SALARY']

# omit value W - statistic "_"
_, p10r = stats.shapiro(rolnictwo_sal)
_, p10p = stats.shapiro(pedagogika_sal)
print(f"Salary normality - Rolnictwo: p = {p10r:.4e}")
print(f"Salary normality - Pedagogika: p = {p10p:.4f}")
print("Brak rozkladu normalnego p = 1.2215 x 10^-9 < 0.05")


# Q - Q plots

def generate_qq_plot(data, title):
    sm.qqplot(data, line='s')
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

# Ex. 6 plot
generate_qq_plot(men_h, "Q-Q Plot: Males height Ex. 6")
generate_qq_plot(women_h, "Q-Q Plot: Women height Ex. 6")

#  Ex. 7
generate_qq_plot(patients['cukier'], "Q-Q Plot: Sugar (glucose) level (Ex. 7)")

# Ex. 8
generate_qq_plot(lightBulbs['czas'], "Q-Q Plot: Light bulbs time (Ex. 8)")

# Ex. 9
generate_qq_plot(capacitors['pojemnosc'], "Q-Q Plot: Capacitors (Ex. 9)")

# Ex. 10
generate_qq_plot(rolnictwo_sal, "Q-Q Plot: Salary - Farming  (Ex. 10)")
generate_qq_plot(pedagogika_sal, "Q-Q Plot: Salary - Pedagogics  (Ex. 10)")

# ECDF plots for Ex. 4, 5, 6

def plot_ks_comparison(data1, data2, label1, label2, title):
    def get_ecdf(data):
        x = np.sort(data)
        y = np.arange(1, len(data) + 1) / len(data)
        return x, y

    x1, y1 = get_ecdf(data1)
    x2, y2 = get_ecdf(data2)

    all_x = np.sort(np.unique(np.concatenate([x1, x2])))
    y1_interp = np.searchsorted(x1, all_x, side='right') / len(x1)
    y2_interp = np.searchsorted(x2, all_x, side='right') / len(x2)
    diffs = np.abs(y1_interp - y2_interp)
    max_idx = np.argmax(diffs)

    x_d = all_x[max_idx]
    y_low = min(y1_interp[max_idx], y2_interp[max_idx])
    y_high = max(y1_interp[max_idx], y2_interp[max_idx])

    plt.figure(figsize=(8, 6))
    plt.step(x1, y1, where='post', color='black', label=label1, linewidth=1)
    plt.step(x2, y2, where='post', color='black', linestyle='--', label=label2, linewidth=1)


    plt.vlines(x_d, y_low, y_high, color='red', linewidth=1.5)
    plt.text(x_d, (y_low + y_high) / 2, ' D', color='red', fontweight='bold', fontstyle='italic')

    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Fraction of total")
    plt.tick_params(direction='in', top=True, right=True)
    plt.legend(loc='lower right')
    plt.ylim(0, 1.05)
    plt.show()




plot_ks_comparison(men_h, women_h, 'Mężczyźni', 'Kobiety',
                   "KS-Test Comparison: Wzrost (Zad. 4 & 6)")

plot_ks_comparison(delikates, renety, 'Delikates', 'Renety',
                   "KS-Test Comparison: Czas na drzewie (Zad. 5)")