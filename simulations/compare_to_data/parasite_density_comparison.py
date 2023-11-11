import os
from functools import partial

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import binom
from statsmodels.stats.proportion import proportion_confint

from create_plots.helpers_reformat_sim_ref_dfs import get_mean_from_upper_age, \
    match_sim_ref_ages, get_age_bin_averages, combine_higher_dens_freqs
from simulations import manifest

from simulations.load_inputs import load_sites
from simulations.helpers import load_coordinator_df

coord_csv = load_coordinator_df(characteristic=False, set_index=True)

density_sites = []
sites = load_sites()
for site in sites:
    if coord_csv.at[site, 'age_parasite_density'] == 1 :
        density_sites.append(site)


print(density_sites)
coord_csv = pd.read_csv(manifest.simulation_coordinator_path)

def prepare_parasite_density_comparison_single_site(sim_df, site):
    """
        Read in, align, and combine reference and simulation data for a single site with a parasite density validation relationship.
        Args:
            sim_df: Simulation data from data-grabbing analyzer.
            Assumed to have the columns something like "sample_number", "Run_Number", ...

        Returns: A dataframe containing the combined reference and simulation data for this validation relationship
        """

    # Load reference data
    # todo: write a common method to generate age_agg_df for sim, ref and benchmark data
    upper_ages = sorted(sim_df['agebin'].unique())
    sim_df['mean_age'] = sim_df['agebin'].apply(get_mean_from_upper_age, upper_ages=upper_ages)
    if "level_1" in sim_df.columns:
        #print("is In")
        age_agg_sim_df = sim_df.groupby("param_set")\
            .apply(get_age_bin_averages)\
            .reset_index() \
            .drop(columns="level_1")
    else:
        #print(" not in")
        age_agg_sim_df = sim_df.groupby("param_set")\
            .apply(get_age_bin_averages)\
            .reset_index()

    filepath_ref = os.path.join(manifest.base_reference_filepath,
                                coord_csv[coord_csv['site'] == site]['age_parasite_density_ref'].iloc[0])
    ref_df = pd.read_csv(filepath_ref)
    ref_df = ref_df[ref_df['Site'].str.lower() == site.lower()]
    ref_df['Site'] = ref_df['Site'].str.lower()
    upper_ages = sorted(ref_df['agebin'].unique())
    ref_df['mean_age'] = ref_df['agebin'].apply(get_mean_from_upper_age, upper_ages=upper_ages)

    # subset simulation output to months in reference dataset
    months = sorted(ref_df['month'].unique())
    print(site)
    print(months)
    print(sim_df)
    print(age_agg_sim_df)
    sim_df = age_agg_sim_df[age_agg_sim_df['month'].isin(months)]

    # if the maximum reference density bin is < (maximum simulation density bin / max_magnitude_difference), aggregate all simulation densities >= max ref bin into the max ref bin
    #   the final bin will be all densities equal to or above that value
    max_ref_dens = ref_df['densitybin'].max(skipna=True)

    combine_higher_dens_freqs_simplified = partial(combine_higher_dens_freqs,
                                                   max_ref_dens=max_ref_dens,
                                                   max_magnitude_difference=100)
    sim_df = sim_df.groupby("param_set")\
        .apply(combine_higher_dens_freqs_simplified)\
        .reset_index()\
        .drop(columns="index")

    # add zeros for unobserved reference densities up to max_ref_dens.
    # Do this by taking the sim dataframe for a single parameter set and using it to add missing zeros in ref_df
    all_zeros_df = sim_df[sim_df["param_set"]==np.min(sim_df["param_set"])].reset_index(drop=True)
    all_zeros_df = all_zeros_df[['month', 'mean_age', 'agebin', 'densitybin', 'Site']]
    ref_df = pd.merge(ref_df, all_zeros_df, how='outer')
    ref_df.fillna(0, inplace=True)

    # fixme - match_sim_ref_ages is currently copied over from the validation framework and it generates an additional
    # fixme - dataframe bench_df that we don't care about right now

    def _match_sim_ref_ages_simple(df):
        #fixme Simple wrapper function since we don't care about bench_df
        return match_sim_ref_ages(ref_df, df)[0]

    sim_df = sim_df.groupby("param_set")\
        .apply(_match_sim_ref_ages_simple)\
        .reset_index()\
        .drop(columns="index")


    # # format reference data
    # if "year" in ref_df.columns:
    #     print(f"A: {site}")
    #     min_yr = np.min(ref_df['year'])
    #     ref_df['year'] = ref_df['year']-min_yr+1
    #     print(ref_df)
    # if not "year" in ref_df.columns:
    #     print(f"B: {site}")
    #     ref_df.insert(0,'year',1)
    #     print(ref_df)
    
    ref_df["site_month"] = ref_df['Site'] + '_month' + ref_df['month'].astype('str')
    #ref_df["site_month"] = ref_df['Site'] + '_year' + ref_df['year'].astype('str') + '_month' + ref_df['month'].astype('str')
    ref_df_asex = ref_df[["asexual_par_dens_freq", "mean_age", "agebin", "densitybin", "Site", "month",
                          "site_month", "bin_total_asex", "count_asex"]] \
        .rename(columns={"asexual_par_dens_freq": "reference",
                         "bin_total_asex": "ref_total",
                         "count_asex": "ref_bin_count"})
    ref_df_gamet = ref_df[["gametocyte_dens_freq", "mean_age", "agebin", "densitybin", "Site", "month",
                           "site_month", "bin_total_gamet", "count_gamet"]] \
        .rename(columns={"gametocyte_dens_freq": "reference",
                         "bin_total_gamet": "ref_total",
                         "count_gamet": "ref_bin_count"})

    # format new simulation output
    # min_yr = np.min(sim_df['year'])
    # sim_df['year'] = sim_df['year']-min_yr+1
    sim_df["site_month"] = sim_df['Site'] + '_month' + sim_df['month'].astype('str')
    #sim_df["site_month"] = sim_df['Site'] + '_year' + sim_df['year'].astype('str') + '_month' + sim_df['month'].astype('str')
    sim_df_asex = sim_df[["param_set", "asexual_par_dens_freq", "mean_age", "agebin", "densitybin", "Site", "month","site_month"]] \
        .rename(columns={"asexual_par_dens_freq": "simulation"})
    sim_df_gamet = sim_df[["param_set", "gametocyte_dens_freq", "mean_age", "agebin", "densitybin", "Site", "month","site_month"]] \
        .rename(columns={"gametocyte_dens_freq": "simulation"})

    # combine reference and simulation dataframes
    combined_df_asex = pd.merge(ref_df_asex, sim_df_asex, how='outer')
    combined_df_asex['metric'] = 'asexual_density'

    combined_df_gamet = pd.merge(ref_df_gamet, sim_df_gamet, how='outer')
    combined_df_gamet['metric'] = 'gametocyte_density'

    return combined_df_asex, combined_df_gamet


def compute_density_likelihood(combined_df):
    """
    Calculate an approximate likelihood for the simulation parameters for each site. This is estimated as the product,
    across age groups, of the probability of observing the reference values if the simulation means represented the
    true population mean
    Args:
        combined_df (): A dataframe containing both the reference and matched simulation output
        sim_column (): The name of the column of combined_df to use as the simulation output
    Returns: A dataframe of loglikelihoods where each row corresponds to a site-month

    """
    #fixme Monique had a different approach in the model-validation framework
    #fixme 230328: JS changed to a simplified approach: naively assume every observation is independent.
    # Likelihood of each observation is likelihood of seeing reference data if simulation is "reality"
    binom_ll = np.vectorize(binom.logpmf) # probability mass function of binomial distribution

    #combined_df.dropna(inplace=True)

    #fixme - Fudge to correct for sim density prevalence values of 1s and 0s (often because of small denominator)
    #fixme - which wreak havoc in likelihoods.  So instead set a range from [0.001,0.999], given typical population size
    #fixme - of 1000 individuals.
    def _correct_extremes(x):
        if x < 0.001:
            return 0.001
        elif x > 0.999:
            return 0.999
        else:
            return x

    combined_df['simulation'] = combined_df['simulation'].apply(_correct_extremes)

    combined_df["ll"] = binom_ll(combined_df["ref_bin_count"],
                                 combined_df["ref_total"],
                                 combined_df["simulation"])
    #print(combined_df)
    return combined_df["ll"].sum()

# The following function determines whether any parameters sets were missing for a site,
# if there are missing parameter set, this prepares compute_LL_by_site to shoot out a warning message
# This function additionally adds the missing parameter set to the dataframe with NaN for the ll.
def identify_missing_parameter_sets(combined_df, numOf_param_sets):
    param_list = list(range(1,numOf_param_sets+1))
    missing_param_sets = []
    for x in param_list:
        if x not in combined_df['param_set'].values:
            combined_df.loc[len(combined_df.index)] = [x,np.NaN]
            missing_param_sets.append(x)
    return combined_df, missing_param_sets
    
def compute_parasite_density_LL_by_site(site, numOf_param_sets):
    sim_df = pd.read_csv(os.path.join(manifest.simulation_output_filepath, site, "parasite_densities_by_age_month.csv"))
    combined_df_asex, combined_df_gamet = prepare_parasite_density_comparison_single_site(sim_df, site)

    asex_LL = combined_df_asex.groupby("param_set")\
        .apply(compute_density_likelihood)\
        .reset_index()\
        .rename(columns={0: "ll"})
        
    asex_LL, missing_param_sets_asex = identify_missing_parameter_sets(asex_LL, numOf_param_sets) 
  
    if len(missing_param_sets_asex) > 0:
        print(f'Warning {site} is missing param_sets {missing_param_sets_asex} for asex parasite density')
              
    asex_LL["metric"] = "asex_density"

    gamet_LL = combined_df_gamet.groupby("param_set")\
        .apply(compute_density_likelihood)\
        .reset_index()\
        .rename(columns={0: "ll"})
   
    gamet_LL, missing_param_sets_gamet = identify_missing_parameter_sets(gamet_LL, numOf_param_sets) 
    
    if len(missing_param_sets_gamet) > 0:
        print(f'Warning {site} is missing param_sets {missing_param_sets_gamet} for gamet parasite density')
        
    gamet_LL["metric"] = "gamet_density"        
    
    return_df = pd.concat([asex_LL, gamet_LL], ignore_index=True)
    
    return_df["site"] = site
        
    return return_df


def compute_parasite_density_LL_for_all_sites(numOf_param_sets):
    df_by_site = []
    for s in density_sites:
        df_this_site = compute_parasite_density_LL_by_site(s, numOf_param_sets)
        df_by_site.append(df_this_site)

    return pd.concat(df_by_site)


def plot_density_comparison_single_site(site, param_sets_to_plot=None, plt_dir=os.path.join(manifest.simulation_output_filepath, "_plots")):
    # Plot comparison for a specific site, given specific param_set
    sim_df = pd.read_csv(os.path.join(manifest.simulation_output_filepath, site, "parasite_densities_by_age_month.csv"))
    combined_df_asex, combined_df_gamet = prepare_parasite_density_comparison_single_site(sim_df, site)

    if param_sets_to_plot is None:
        param_sets_to_plot = list(set(combined_df_asex["param_set"]))

    #fixme hack
    # combined_df_asex = combined_df_asex[combined_df_asex["param_set"]==1]
    # combined_df_gamet = combined_df_gamet[combined_df_gamet["param_set"]==1]

    def _plot_parasite_type(parasite_type):
        if parasite_type == "asex":
            df = combined_df_asex
        elif parasite_type == "gamet":
            df = combined_df_gamet
        else:
            raise NotImplemented

        # Shift 0-density to small value to show plots on log scale
        df["densitybin"][df["densitybin"]==0] = 5
        ages = sorted(list(set(df["mean_age"])))

        plt.figure(figsize=(5,10))

        foo = df.groupby(["mean_age", "densitybin", "param_set"])\
            .agg({"ref_total": "sum",
                  "ref_bin_count": "sum",
                  "simulation": "mean"})\
            .reset_index()

        _confidence_interval_vectorized = np.vectorize(partial(proportion_confint, method="wilson"))
        foo["reference"] = foo["ref_bin_count"]/foo["ref_total"]
        foo["reference_low"], foo["reference_high"] = _confidence_interval_vectorized(foo["ref_bin_count"],
                                                                                      foo["ref_total"])

        i = 1
        for a, age_df in foo.groupby("mean_age"):
            plt.subplot(len(ages), 1, i)

            have_plotted_ref = False
            for p, param_df in age_df.groupby("param_set"):

                if p in param_sets_to_plot:
                    if not have_plotted_ref:
                        plt.plot(param_df["densitybin"], param_df["reference"], label="ref", marker='o')
                        plt.fill_between(param_df["densitybin"],
                                         param_df["reference_low"],
                                         param_df["reference_high"],
                                         label="95% confidence", alpha=0.2)
                        have_plotted_ref = True

                    plt.plot(param_df["densitybin"], param_df["simulation"], label=f"Param set {p}", marker='o')
                    plt.xscale("log")

                    n_ref_total = param_df["ref_total"].iloc[0]
                    plt.title(f"Mean age = {a}.  Total N = {n_ref_total}")
                    plt.ylim([0,1])
                    plt.xlim([1,1e7])

            plt.xlabel("Parasite density")
            plt.ylabel("Fraction")
            plt.legend(fontsize=11)
            i += 1

        plt.suptitle(f"{site} - {parasite_type}")
        plt.tight_layout()
        #plt.savefig(os.path.join(manifest.simulation_output_filepath, "_plots", f"density_{parasite_type}_{site}.png"))
        plt.savefig(os.path.join(plt_dir,f"density_{parasite_type}_{site}.png"))

    for parasite_type in ["asex", "gamet"]:
        _plot_parasite_type(parasite_type)

def plot_density_comparison_all_sites(param_sets_to_plot=None,plt_dir=os.path.join(manifest.simulation_output_filepath, "_plots")):
    for s in density_sites:
        plot_density_comparison_single_site(s,param_sets_to_plot=param_sets_to_plot,plt_dir=plt_dir)

if __name__ == "__main__":
    #plot_density_comparison_all_sites()
    print(compute_parasite_density_LL_for_all_sites())
