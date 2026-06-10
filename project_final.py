import os
from glob import glob

import fitdecode
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from statsmodels.stats.multicomp import pairwise_tukeyhsd


sns.set_theme(style="whitegrid")


# przygotowanie danych, ekstrakcja, usuwanie dni bez danych HRV i snu

def analyze_fit_folder(folder_path):
    compiled_data = {}
    fit_files = glob(os.path.join(folder_path, "*.fit"))
    # getting date, HRV, sleep length
    for file_path in fit_files:
        file_date = None
        hrv_avg = None
        sleep_timestamps = []

        try:
            with fitdecode.FitReader(file_path) as fit:
                for frame in fit:
                    if isinstance(frame, fitdecode.FitDataMessage):
                        if frame.name == 'file_id':
                            for field in frame.fields:
                                if field.name == 'time_created' and field.value is not None:
                                    file_date = field.value.date()

                        if frame.name == 'sleep_level':
                            fields = {f.name: f.value for f in frame.fields}
                            if 'timestamp' in fields and fields['timestamp'] is not None:
                                sleep_timestamps.append(fields['timestamp'])

                        if frame.name == 'hrv_status_summary':
                            fields = {f.name: f.value for f in frame.fields}
                            hrv_avg = fields.get('last_night_average')

            if file_date:
                if file_date not in compiled_data:
                    compiled_data[file_date] = {'Sleep_Duration_Hours': None, 'HRV_Avg': None}

                if sleep_timestamps:
                    sleep_delta = max(sleep_timestamps) - min(sleep_timestamps)
                    sleep_hours = round(sleep_delta.total_seconds() / 3600, 2)
                    prev_sleep = compiled_data[file_date]['Sleep_Duration_Hours']
                    compiled_data[file_date]['Sleep_Duration_Hours'] = max(sleep_hours, prev_sleep if prev_sleep else 0)

                if hrv_avg:
                    compiled_data[file_date]['HRV_Avg'] = hrv_avg

        except Exception as e:
            print(f"Błąd pliku {os.path.basename(file_path)}: {e}")

    df_health = pd.DataFrame.from_dict(compiled_data, orient='index').reset_index()
    df_health.rename(columns={'index': 'Merge_Date'}, inplace=True)
    return df_health



if __name__ == "__main__":
    csv_file = "data/project/activities_running.csv"
    wellness_folder = "data/project/wellness/"


    df_runs = pd.read_csv(csv_file)
    df_runs['Date'] = pd.to_datetime(df_runs['Date'])
    df_runs['Merge_Date'] = df_runs['Date'].dt.date
    # check if Avg HR is a number
    df_runs['Avg HR'] = pd.to_numeric(df_runs['Avg HR'], errors='coerce')

    df_health = analyze_fit_folder(wellness_folder)

    # same types before merge
    df_runs['Merge_Date'] = pd.to_datetime(df_runs['Merge_Date']).dt.date
    df_health['Merge_Date'] = pd.to_datetime(df_health['Merge_Date']).dt.date

    # run and health data merge into one table
    final_df = pd.merge(df_runs, df_health, on='Merge_Date', how='left')


    # reject days without Avg HR, sleep, HRV
    final_df = final_df.dropna(subset=['Avg HR', 'Sleep_Duration_Hours', 'HRV_Avg']).copy()





    # Ex. 1
    # Define 3 populations (criterion: sleep length)
    # Testing attribute Avg HR during run

    def assign_population(sleep_hours):
        if sleep_hours < 6.5:
            return 'Krotki_Sen'
        elif 6.5 <= sleep_hours <= 8.0:
            return 'Optymalny_Sen'
        else:
            return 'Dlugi_Sen'


    final_df['Population'] = final_df['Sleep_Duration_Hours'].apply(assign_population)


    # Dividing to populations for tests
    pop1 = final_df[final_df['Population'] == 'Krotki_Sen']['Avg HR'].dropna().values
    pop2 = final_df[final_df['Population'] == 'Optymalny_Sen']['Avg HR'].dropna().values
    pop3 = final_df[final_df['Population'] == 'Dlugi_Sen']['Avg HR'].dropna().values

    print(f"Liczebność populacji - Krótki sen: {len(pop1)}, Optymalny: {len(pop2)}, Długi: {len(pop3)}")

    # Ex. 2 descriptive statistics
    print("\n cw.2 Statystyki opisowe dla 'Avg HR' w populacjach:")
    stats_desc = final_df.groupby('Population')['Avg HR'].describe(percentiles=[.25, .5, .75])

    print(stats_desc[['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']])


    plt.figure(figsize=(8, 5))
    # sns.boxplot(x='Population', y='Avg HR', data=final_df, palette="Set2")
    sns.boxplot(x='Population', y='Avg HR', hue='Population', data=final_df, palette="Set2", legend=False)
    plt.title("Rozkład Średniego Tętna w zależności od długości snu")
    plt.xlabel("Populacja (Grupa snu)")
    plt.ylabel("Średnie Tętno z biegu (Avg HR)")
    plt.savefig("cw2_boxplot.png")
    plt.close()


    # Ex. 3 Global test of the equality of a attribute in all populations

    print("\n cw. 3: weryfikacja zalozen i test globalny")

    # Zmienne są niepowiazane (niezależne) – bo bieg byl w różne dni

    # 3 checking normal distribuiton (Shapiro-Wilka)
    print("test Shapiro-Wilka (p-value):")
    p_shapiro = [stats.shapiro(p)[1] if len(p) >= 3 else np.nan for p in [pop1, pop2, pop3]]
    print(f"  Krótki: {p_shapiro[0]:.4f}, Optymalny: {p_shapiro[1]:.4f}, Długi: {p_shapiro[2]:.4f}")

    # 3 Sprawdzenie równości wariancji (Test Levene'a)
    stat_lev, p_levene = stats.levene(pop1, pop2, pop3)
    print(f"- Test równości wariancji Levene'a: p-value = {p_levene:.4f}")

    # Wybór testu głównego na podstawie założeń
    if all(p > 0.05 for p in p_shapiro) and p_levene > 0.05:
        # ANOVA parametryczna
        f_stat, p_global = stats.f_oneway(pop1, pop2, pop3)
        print(f"-> Rozkłady normalne, wariancje równe. Test ANOVA. p-value = {p_global:.4f}")
        test_type = "ANOVA"
    else:
        # Kruskal-Wallis nieparametryczny
        h_stat, p_global = stats.kruskal(pop1, pop2, pop3)
        print(f"-> Założenia niepselnione. Nieparametryczny test Kruskala-Wallisa. p-value = {p_global:.4f}")
        test_type = "Kruskal-Wallis"

    # Analiza Post-Hoc (jeśli p_global < 0.05)
    if p_global < 0.05:
        print("Sa istotne roznice. Analiza Post-Hoc (Tukey HSD)...")
        tukey = pairwise_tukeyhsd(endog=final_df['Avg HR'], groups=final_df['Population'], alpha=0.05)
        print(tukey)
    else:
        print("- Brak podstaw do odrzucenia H0: Populacje mają zbliżone średnie/mediany.")


    # Ex. 4 porównania parami (największa średnia / mediana)

    print("\n Ex.4: Porównania parami (Testy T / Mann-Whitney)")

    pairs = [("Krótki vs Optymalny", pop1, pop2),
             ("Krótki vs Długi", pop1, pop3),
             ("Optymalny vs Długi", pop2, pop3)]

    for name, p_a, p_b in pairs:
        if test_type == "ANOVA":
            stat, p_val = stats.ttest_ind(p_a, p_b, equal_var=True)
            print(f"  {name}  Test t-Studenta p-value = {p_val:.4f}")
        else:
            stat, p_val = stats.mannwhitneyu(p_a, p_b, alternative='two-sided')
            print(f"  {name}  Test Manna-Whitneya p-value = {p_val:.4f}")



    # Ex. 5
    print("\nTesty jednostronne (Czy tetno grupy < Próg +10%)")

    global_mean = final_df['Avg HR'].mean()
    cutoff_value = global_mean * 1.10
    print(f"  Ogólna średnia tętna: {global_mean:.2f} ud/min. Próg (+10%): {cutoff_value:.2f} ud/min")

    populations_dict = {'Krótki_Sen': pop1, 'Optymalny_Sen': pop2, 'Długi_Sen': pop3}

    for name, data in populations_dict.items():
        if len(data) > 1:
            # H0: Średnia >= cutoff_value vs H1: Średnia < cutoff_value
            stat, p_val = stats.ttest_1samp(data, popmean=cutoff_value, alternative='less')
            print(f"  {name}: p-value jednostronnego testu t = {p_val:.4f}")
        else:
            print(f"  {name}: Zbyt mała próba do testu.")

    #  Ex. 6 przedz ufn.
    print("\nEx. 6: Przedziały Ufności dla Średniej i Wariancji")

    for name, data in populations_dict.items():
        n = len(data)
        if n < 2:
            print(f"  {name}: Populacja za mala, dla wyznaczenia przedzialow.")
            continue

        mean = np.mean(data)
        sem = stats.sem(data)
        var = np.var(data, ddof=1)

        # przedzialy dla sredniej (Rozkl. t-Studenta)
        ci_mean_95 = stats.t.interval(0.95, df=n - 1, loc=mean, scale=sem)
        ci_mean_99 = stats.t.interval(0.99, df=n - 1, loc=mean, scale=sem)

        # Przedzialy dla wariancji (rozkl. Chi-Kwadrat)
        chi2_lower_95 = stats.chi2.ppf(0.025, df=n - 1)
        chi2_upper_95 = stats.chi2.ppf(0.975, df=n - 1)
        ci_var_95 = ((n - 1) * var / chi2_upper_95, (n - 1) * var / chi2_lower_95)

        chi2_lower_99 = stats.chi2.ppf(0.005, df=n - 1)
        chi2_upper_99 = stats.chi2.ppf(0.995, df=n - 1)
        ci_var_99 = ((n - 1) * var / chi2_upper_99, (n - 1) * var / chi2_lower_99)

        print(f"\n  * POPULACJA: {name} (n={n})")
        print(
            f"    Średnia: {mean:.2f} | 95% CI: ({ci_mean_95[0]:.2f}, {ci_mean_95[1]:.2f}) | 99% CI: ({ci_mean_99[0]:.2f}, {ci_mean_99[1]:.2f})")
        print(
            f"    Wariancja: {var:.2f} | 95% CI: ({ci_var_95[0]:.2f}, {ci_var_95[1]:.2f}) | 99% CI: ({ci_var_99[0]:.2f}, {ci_var_99[1]:.2f})")

