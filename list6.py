import numpy as np
import scipy.io as sio
import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import statsmodels.api as sm
from statsmodels.formula.api import ols


def load_file(path='data/anova_data.mat'):
    data = sio.loadmat(path)
    return data


def ex1_coala_sleep(dane):
    print("Ex. 1 Koala sen")


    koala = dane['koala']
    group1 = koala[:, 0]
    group2 = koala[:, 1]
    group3 = koala[:, 2]
    groups = [group1, group2, group3]

    alpha = 0.05

    print("\nSprawdzanie normalności rozkładu Shapiro-Wilk")
    print("H0: Próbka z rozkładu normalnego.")
    print("H1: Próbka nie z rozkładu normalnego")

    isDistrNormal = True
    for i, group in enumerate(groups, 1):
        stat, p_val = stats.shapiro(group)
        print(f"Grupa {i}: p-value = {p_val:.4f}", end=" -> ")
        if p_val > alpha:
            print("Brak podstaw do odrzucenia H0 (rozkład uznajemy za normalny).")
        else:
            print("Odrzucamy H0 (rozkład nie jest normalny).")
            isDistrNormal = False

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Q-Q Plot dla grup koali')
    for i, group in enumerate(groups):
        stats.probplot(group, dist="norm", plot=axes[i])
        axes[i].set_title(f'Grupa {i + 1}')
    plt.tight_layout()
    plt.show()

    print("\nSprawdzenie równości wariancji - Test Levene'a")
    print("H0: Wariancje we wszystkich grupach są równe.")
    print("H1: Conajmnniej jedna wariancja różni się od pozostałych.")

    stat_lev, p_val_lev = stats.levene(group1, group2, group3)
    print(f"Test Levene'a: p-value = {p_val_lev:.4f}", end=" -> ")

    isVarianceSame = True
    if p_val_lev > alpha:
        print("Brak podstaw do odrzucenia H0 (wariancje równe).")
    else:
        print("Odrzucamy H0 (wariancje są różne).")
        isVarianceSame = False

    if isDistrNormal and isVarianceSame:
        print("\nJednoczynnikowa Analiza Wariancji (ANOVA)")
        print("H0: Średnie we wszystkich populacjach są równe (mu1 = mu2 = mu3).")
        print("H1: Przynajmniej jedna średnia różni się od pozostałych.")

        stat_f, p_val_f = stats.f_oneway(group1, group2, group3)
        print(f"ANOVA (Fishera): F-statistic = {stat_f:.4f}, p-value = {p_val_f:.4e}")

        if p_val_f < alpha:
            print("Wniosek: Odrzucamy H0. Czas snu koali znacząco różni się między badanymi grupami.")
        else:
            print(
                "Wniosek: Brak podstaw do odrzucenia H0. Nie ma dowodów na to, że średni czas snu różni się między grupami.")
    else:
        print("\nZałożenia dla klasycznej ANOVY (normalność lub równość wariancji) nie są spełnione.")



def ex_2_wombats(dane):
    print("\nEx. 2: Aktywność dobowa wombatow")


    wombats = dane['wombats'].flatten()
    wombats_groups = dane['wombat_groups'].flatten()


    groups_data = [wombats[wombats_groups == g] for g in np.unique(wombats_groups)]

    print("\nZalozenia: rozkłady są normalne i wariancje równe (wedlug polecenia).")

    print("\nJednoczynnikowa Analiza Wariancji (ANOVA)")
    print("H0: Średnie we wszystkich populacjach wombatów są równe.")
    print("H1: Przynajmniej jedna średnia różni się od pozostałych.")

    stat_f, p_val_f = stats.f_oneway(*groups_data)

    print(f"ANOVA (Fishera): F-statistic = {stat_f:.4f}, p-value = {p_val_f:.4e}")

    alpha = 0.05
    if p_val_f < alpha:
        print("Wniosek: P-value jest mniejsze niż założony poziom istotności (0.05).")
        print(
            "Odrzucamy hipotezę zerową (H0). Istnieją statystycznie istotne różnice w aktywności dobowej wombatów między poszczególnymi grupami populacji.")
    else:
        print("Wniosek: P-value jest większe niż założony poziom istotności (0.05).")
        print(
            "Brak podstaw do odrzucenia hipotezy zerowej. Nie ma dowodów na istotne różnice w średniej aktywności w badanych populacjach.")



def ex3_candy_bar():
    print("\nEx. 3: Wpływ kampanii na sprzedaż batonu")

    q1 = np.array([3415, 1593, 1976, 1526, 1538, 983, 1050, 1861, 1714, 1320, 1276, 1263, 1271, 1436])
    q2 = np.array([4556, 1937, 2056, 1594, 1634, 1086, 1209, 2087, 2415, 1621, 1377, 1279, 1417, 1310])
    q3 = np.array([5772, 2242, 2240, 1644, 1866, 1135, 1245, 2054, 2361, 1624, 1522, 1350, 1583, 1357])
    q4 = np.array([5432, 2794, 2085, 1705, 1769, 1177, 977, 2018, 2424, 1551, 1412, 1490, 1513, 1468])

    quarters = [q1, q2, q3, q4]
    names = ["Kwartał I", "Kwartał II", "Kwartał III", "Kwartał IV"]
    alpha = 0.05

    print("\nSprawdzanie normalności rozkładu Shapiro-Wilk")
    print("H0: Próbka w danym kwartale pochodzi z populacji o rozkładzie normalnym.")

    isDistrNormal = True
    for i, (quarters_data, name) in enumerate(zip(quarters, names)):
        stat, p_val = stats.shapiro(quarters_data)
        print(f"{name}: p-value = {p_val:.4f}", end=" ")
        if p_val > alpha:
            print("Brak podstaw do odrzucenia H0 (rozkład można uznać za normalny).")
        else:
            print("Odrzucamy H0 (rozkład nie jest normalny).")
            isDistrNormal = False

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle('Q-Q Plot dla poszczególnych kwartałów')
    for i, (quarters_data, name) in enumerate(zip(quarters, names)):
        stats.probplot(quarters_data, dist="norm", plot=axes[i])
        axes[i].set_title(name)
    plt.tight_layout()
    plt.show()

    # we compare same shops so we have dependent samples
    if isDistrNormal:
        print("Rozkłady są normalne. Mozna Anova, ale polecany w liscie jest Friedmana")
    else:
        print("Założenie o normalności nie jest spełnione dla wszystkich kwartałów.")
        # print(" Robimy nieparametryczny test Friedmana.")

    print("\nTest Friedmana")
    print("H0: Mediany wielkości sprzedaży we wszystkich kwartałach są równe (kampania nie miała wpływu).")
    print("H1: Przynajmniej jedna mediana różni się od pozostałych.")

    stat_friedman, p_val_friedman = stats.friedmanchisquare(q1, q2, q3, q4)
    print(f"Statystyka testowa = {stat_friedman:.4f}, p-value = {p_val_friedman:.4e}")

    if p_val_friedman < alpha:
        print("\nWynik: p-value jest mniejsze niż 0.05. Odrzucamy H0.")
        print(
            "Wniosek: Istnieją statystycznie istotne różnice w wielkości sprzedaży między poszczególnymi kwartałami.")
        print("Kampania reklamowa najprawdopodobniej miała wpływ na wielkość sprzedaży reklamowanego batonu.")
    else:
        print("\nWynik: p-value jest większe niż 0.05. Brak podstaw do odrzucenia H0.")
        print("Wniosek: Nie ma dowodów na to, by sprzedaż różniła się istotnie między kwartałami.")


def ex_4_popcorn():

    print("\nEx. 4: Analiza post-hoc dla danych 'popcorn'")


    # Columns: manufacturer  1, 2 and 3
    # Rows 1-3: Air device
    # Rows 4-6: Oil device
    popcorn = np.array([
        [5.5, 4.5, 3.5],
        [5.5, 4.5, 4.0],
        [6.0, 4.0, 3.0],
        [6.5, 5.0, 4.0],
        [7.0, 5.5, 5.0],
        [7.0, 5.0, 4.5]
    ])


    efficiency = popcorn.flatten()


    manufacturer = np.tile(['Prod_1', 'Prod_2', 'Prod_3'], 6)


    device = np.repeat(['Powietrzna', 'Olejowa'], 9)


    df = pd.DataFrame({
        'Wydajnosc': efficiency,
        'Producent': manufacturer,
        'Maszyna': device
    })

    alpha = 0.05

    print("\nTest post-hoc Tukeya dla czynnika: producent")
    tukey_producent = pairwise_tukeyhsd(endog=df['Wydajnosc'], groups=df['Producent'], alpha=alpha)
    print(tukey_producent)

    print("\nTest post-hoc Tukeya dla czynnika: typ maszyny")
    tukey_maszyna = pairwise_tukeyhsd(endog=df['Wydajnosc'], groups=df['Maszyna'], alpha=alpha)
    print(tukey_maszyna)

    # Wizualizacja przedziałów ufności dla różnic odp. z matlab
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    tukey_producent.plot_simultaneous(ax=axes[0], xlabel='Wydajność (liczba kubków)', ylabel='Producent')
    axes[0].set_title('Porównanie średnich - Producent')

    tukey_maszyna.plot_simultaneous(ax=axes[1], xlabel='Wydajność (liczba kubków)', ylabel='Maszyna')
    axes[1].set_title('Porównanie średnich - typ maszyny')

    plt.tight_layout()
    plt.show()

    print("\n[ Wnioski z an. post-hoc ]")
    print("1. producent 1 vs producent 2 p-value < 0.05 i reject = true. Istnieje istotna różnica – producent 1 daje znacząco więcej popcornu niż producent 2.")
    print("    producent 1 vs producent 3 p-value < 0.05 i reject = true. Istnieje istotna różnica – producent 1 daje znacząco więcej popcornu niż producent 3.")
    print("   producent 2 vs producent 3 p-value > 0.05 i reject = false. Brak istotnej statystycznie różnicy między Producentem 2 i 3.")
    print("   Wniosek: kukurydza od producenta 1 jest najlepsza. kukurydza od prod. 2 i 3 daja statystycznie podobne rezultaty")

    print("2. Typ maszyny: p-value > 0.05 i reject = False")
    print("   Zgodnie z testem Tukeya nie ma istotnej statystycznie różnicy w wydajności między maszyną olejową a powietrzną. Bardzo malo braklo aby zejsc < 0.05")
    print("W Matlabie jest multcompare, który ma większą moc, niż Turkeya w pythonie i wyniki mogą się różnić miedzyh pythonem a matlabem")


def ex_5_toxic():
    print("\nEx. 5: Wpływ substancji toksycznych i zakładu na FEV")



    raw_data = [
        [4.64, 5.12, 4.64, 3.21, 3.92, 4.95, 3.75, 2.95, 2.95],
        [5.92, 6.10, 4.32, 3.17, 3.75, 5.22, 2.50, 3.21, 2.80],
        [5.25, 4.85, 4.13, 3.88, 4.01, 5.16, 2.65, 3.15, 3.63],
        [6.17, 4.72, 5.17, 3.50, 4.64, 5.35, 2.84, 3.25, 3.85],
        [4.20, 5.36, 3.77, 2.47, 3.63, 4.35, 3.09, 2.30, 2.19],
        [5.90, 5.41, 3.85, 4.12, 3.46, 4.89, 2.90, 2.76, 3.32],
        [5.07, 5.31, 4.12, 3.51, 4.01, 5.61, 2.62, 3.01, 2.68],
        [4.13, 4.78, 5.07, 3.85, 3.39, 4.98, 2.75, 2.31, 3.35],
        [4.07, 5.08, 3.25, 4.22, 3.78, 5.77, 3.10, 2.50, 3.12],
        [5.30, 4.97, 3.49, 3.07, 3.51, 5.23, 1.99, 2.02, 4.11],
        [4.37, 5.85, 3.65, 3.62, 3.19, 4.76, 2.42, 2.64, 2.90],
        [3.76, 5.26, 4.10, 2.95, 4.04, 5.15, 2.37, 2.27, 2.75]
    ]
    raw_data = np.array(raw_data)

    substances = ['T1', 'T1', 'T1', 'T2', 'T2', 'T2', 'T3', 'T3', 'T3']
    plant = ['Z1', 'Z2', 'Z3', 'Z1', 'Z2', 'Z3', 'Z1', 'Z2', 'Z3']


    dane_lista = []
    for col_idx in range(9):
        subst = substances[col_idx]
        zaklad = plant[col_idx]
        for val in raw_data[:, col_idx]:
            dane_lista.append({'FEV': val, 'Substancja': subst, 'Zaklad': zaklad})

    df = pd.DataFrame(dane_lista)

    alpha = 0.05
    groups = df.groupby(['Substancja', 'Zaklad'])

    print("\nSprawdzenie normalności rozkładu w podgrupach Shapiro-Wilk")
    isDistrNormal = True
    for name, group in groups:
        stat, p_val = stats.shapiro(group['FEV'])
        print(f"Grupa {name[0]}-{name[1]}: p-value = {p_val:.4f}", end="")
        if p_val > alpha:
            print("OK (rozkład normalny)")
        else:
            print("Odrzucamy H0 (brak normalności!)")
            isDistrNormal = False


    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    fig.suptitle('Q-Q Plot dla każdej z podgrup (Substancja - Zakład)')

    for ax, (name, group) in zip(axes.flatten(), groups):
        stats.probplot(group['FEV'], dist="norm", plot=ax)
        ax.set_title(f'{name[0]} - {name[1]}')

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.show()

    print("\nSprawdzenie jednorodności wariancji Test Levene'a")
    lista_danych_grup = [grupa['FEV'].values for nazwa, grupa in groups]
    stat_lev, p_val_lev = stats.levene(*lista_danych_grup)

    print(f"Test Levene'a: p-value = {p_val_lev:.4f}", end="")
    if p_val_lev > alpha:
        print("OK (Brak podstaw do odrzucenia H0 - wariancje są równe)")
    else:
        print("Odrzucamy H0 (wariancje są różne)")

    print("\nDwuczynnikowa Analiza Wariancji Two-Way ANOVA")
    print("hipotezy zerowe:")
    print("H01: Średnie FEV wyznaczone względem substancji są równe.")
    print("H02: Średnie FEV wyznaczone względem zakładu są równe.")
    print("H03: Substancja i zakład nie mają synergicznego wpływu na średnie (brak interakcji).")


    model = ols('FEV ~ C(Substancja) + C(Zaklad) + C(Substancja):C(Zaklad)', data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)

    print("\nTabela ANOVA:\n")
    print(anova_table)

    print("\nWnioski z 2 czynnikowej ANOVA")


    p_subst = anova_table.loc['C(Substancja)', 'PR(>F)']
    if p_subst < alpha:
        print(
            f"1. Czynnik Substancja (p = {p_subst:.4e}): Odrzucamy H01. Różne substancje toksyczne w istotny sposób wpływają na zróżnicowanie pojemności oddechowej FEV.")
    else:
        print(
            f"1. Czynnik Substancja (p = {p_subst:.4f}): Brak podstaw do odrzucenia H01. Substancja nie wpływa istotnie na FEV.")


    p_plant = anova_table.loc['C(Zaklad)', 'PR(>F)']
    if p_plant < alpha:
        print(
            f"2. Czynnik Zakład (p = {p_plant:.4e}): Odrzucamy H02. FEV istotnie różni się w zależności od tego, w którym z trzech zakładów pracują pracownicy.")
    else:
        print(f"2. Czynnik Zakład (p = {p_plant:.4f}): Brak podstaw do odrzucenia H02.")

    # Interaction analysis (synergy)
    p_interaction = anova_table.loc['C(Substancja):C(Zaklad)', 'PR(>F)']
    if p_interaction < alpha:
        print(
            f"3. Interakcja (p = {p_interaction:.4e}): Odrzucamy H03. Istnieje synergiczny efekt substancji i zakładu (wpływ substancji na układ oddechowy zależy od tego, w którym zakładzie przebywa pracownik).")
    else:
        print(
            f"3. Interakcja (p = {p_interaction:.4f}): Brak podstaw do odrzucenia H03. Nie ma istotnego efektu współdziałania pomiędzy zakładem a typem substancji.")

if __name__ == "__main__":
    try:
        dane_mat = load_file('data/anova_data.mat')
        ex1_coala_sleep(dane_mat)
        ex_2_wombats(dane_mat)
        ex3_candy_bar()
        ex_4_popcorn()
        ex_5_toxic()
    except FileNotFoundError:
        print("Błąd. Nie znaleziono pliku")