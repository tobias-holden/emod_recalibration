import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
sns.set_context("talk")
sns.set_style("white")
from scipy.stats import binom

from create_plots.helpers_reformat_sim_ref_dfs import get_fraction_in_infectious_bin
from simulations import manifest
from simulations.load_inputs import load_sites
from simulations.helpers import load_coordinator_df

coord_csv = load_coordinator_df(characteristic=False, set_index=True)

infectiousness_sites = []
sites = load_sites()
for site in sites:
    if coord_csv.at[site, 'infectiousness_to_mosquitos'] == 1 :
        infectiousness_sites.append(site)
print("Infectiousness:") 
print(infectiousness_sites)
coord_csv = pd.read_csv(manifest.simulation_coordinator_path)


def prepare_infectiousness_comparison_single_site(sim_df, site):
    """
        Read in, align, and combine reference and simulation data for a single site with an prevalence-by-age validation relationship.
        Args:
            sim_df: Simulation data from data-grabbing analyzer.
            Assumed to have the columns something like "sample_number", "Run_Number", ...

        Returns: A dataframe containing the combined reference and simulation data for this validation relationship
        """

    # Process simulation data
    #fixme sample_number column will be renamed based on upstream analyzer output


    # Load reference data
    filepath_ref = os.path.join(manifest.base_reference_filepath,
                                coord_csv[coord_csv['site'] == site]['infectiousness_to_mosquitos_ref'].iloc[0])
    ref_df = pd.read_csv(filepath_ref)
    ref_df.rename(columns={'site': 'Site'}, inplace=True)
    ref_df = ref_df[ref_df['Site'].str.lower() == str(site).lower()]
    ref_df['Site'] = ref_df['Site'].str.lower()
    ref_months = ref_df['month'].unique()
    print(ref_df)
    # remove simulation rows with zero pop
    sim_df = sim_df[sim_df['Pop'] > 0]
    min_yr = np.min(sim_df['year'])
    sim_df['year'] = sim_df['year']-min_yr+1
        
        
    # subset simulation to months in reference df
    sim_df = sim_df[sim_df['month'].isin(ref_months)]
    # get mean (across simulation seeds and ages within a bin) fraction of individuals in each  {age bin, month,
    # densitybin, run number} group that fall in each infectiousness bin
    # sim_df_agg2 = get_fraction_in_infectious_bin(sim_df)
    if "level_1" in sim_df.columns:
        sim_df_by_param_set = sim_df\
            .groupby("param_set")\
            .apply(get_fraction_in_infectious_bin)\
            .reset_index()\
            .drop(columns="level_1")
    else:
        sim_df_by_param_set = sim_df\
            .groupby("param_set")\
            .apply(get_fraction_in_infectious_bin)\
            .reset_index()
        
    # standardize column names and merge simulation and reference data frames
    min_yr = np.min(ref_df['year'])
    ref_df['year'] = ref_df['year']-min_yr+1
    ref_df["site_month"] = ref_df['Site'] + '_year' + ref_df['year'].astype('str') + '_month' + ref_df['month'].astype('str')
    #ref_df["site_month"] = ref_df['Site'] + '_month' + ref_df['month'].astype('str')
    ref_df = ref_df[["freq_frac_infect", "agebin", "densitybin", f"raction_infected_bin", "Site", "month", "site_month",
                     "num_in_group", "count"]]
    ref_df.rename(columns={"freq_frac_infect": "reference",
                           "num_in_group": "ref_total",
                           "count": "ref_bin_count"},
                  inplace=True)

    sim_df_by_param_set["site_month"] = sim_df_by_param_set['Site'] + '_year' + sim_df_by_param_set['year'].astype('str') + '_month' + sim_df_by_param_set['month'].astype('str')
    #sim_df_by_param_set["site_month"] = sim_df_by_param_set['Site'] + '_month' + sim_df_by_param_set['month'].astype('str')
    sim_df_by_param_set.rename(columns={"infectiousness_bin_freq": "simulation",
                                        "infectiousness_bin": "fraction_infected_bin"},
                               inplace=True)
  
    ref_df['agebin'] = [int(x) for x in ref_df['agebin']]
    sim_df_by_param_set['agebin'] = [int(x) for x in sim_df_by_param_set['agebin']]
    print(ref_df)
    print(sim_df_by_param_set)
    #print(ref_df.columns)
    #print(sim_df_by_param_set.columns)
    #fixme - Note we are dropping nans in both reference and simulation.  Dropping nans in simulation might not be best
    combined_df = pd.merge(sim_df_by_param_set, ref_df, how='inner')#.dropna(subset=["reference"])#, "simulation"])
    combined_df['metric'] = 'infectiousness'
    print(combined_df)
    #fixme - Fudge to correct for sim infectiousness values of 1s and 0s (often because of small denominator)
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

    return combined_df


def compute_infectiousness_likelihood(combined_df):
    """
    Calculate an approximate likelihood for the simulation parameters for each site. This is estimated as the product,
    across age groups, of the probability of observing the reference values if the simulation means represented the
    true population mean
    Args:
        combined_df (): A dataframe containing both the reference and matched simulation output
        sim_column (): The name of the column of combined_df to use as the simulation output
    Returns: A dataframe of loglikelihoods where each row corresponds to a site-month

    """

    #fixme Jaline and Prashanth used dirichlet_multinomial for infectiousness
    #fixme 230328: JS changed to a simplified approach: naively assume every observation is independent.
    # Likelihood of each observation is likelihood of seeing reference data if simulation is "reality"
    binom_ll = np.vectorize(binom.logpmf) # probability mass function of binomial distribution

    combined_df["ll"] = binom_ll(combined_df["ref_bin_count"],
                                 combined_df["ref_total"],
                                 combined_df["simulation"])
    #print(combined_df)
    return combined_df["ll"].sum()#mean()#

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
    
def compute_infectiousness_LL_by_site(site,numOf_param_sets):
    print(site)
    sim_df = pd.read_csv(os.path.join(manifest.simulation_output_filepath, site, "infectiousness_by_age_density_month.csv"))
    #print(sim_df)
    combined_df = prepare_infectiousness_comparison_single_site(sim_df, site)
    #print(combined_df)
    ll_by_param_set = combined_df.groupby("param_set") \
        .apply(compute_infectiousness_likelihood) \
        .reset_index() \
        .rename(columns={0: "ll"})

    ll_by_param_set, missing_param_sets = identify_missing_parameter_sets(ll_by_param_set, numOf_param_sets)
    
    ll_by_param_set["metric"] = "infectiousness"
    ll_by_param_set["site"] = site
    
    if len(missing_param_sets) > 0:
        print(f'Warning {site} is missing param_sets {missing_param_sets} for infectiousness')
    return ll_by_param_set


def compute_infectious_LL_for_all_sites(numOf_param_sets):
    df_by_site = []
    for s in infectiousness_sites:
        df_this_site = compute_infectiousness_LL_by_site(s, numOf_param_sets)
        df_by_site.append(df_this_site)

    return pd.concat(df_by_site)


def plot_infectiousness_comparison_single_site(site, param_sets_to_plot=None,plt_dir=os.path.join(manifest.simulation_output_filepath, "_plots")):
    # Plot comparison for a specific site, given specific param_set
    sim_df = pd.read_csv(os.path.join(manifest.simulation_output_filepath, site, "infectiousness_by_age_density_month.csv"))
    combined_df = prepare_infectiousness_comparison_single_site(sim_df, site)

    #fixme hack
    combined_df = combined_df[combined_df["param_set"] == 1]

    if param_sets_to_plot is None:
        param_sets_to_plot = list(set(combined_df["param_set"]))

    #todo Add error bars on data

    plt.figure(dpi=200, figsize=(15,5))
    # color of lines = month
    # each plot infectiousness vs density
    # subplots span frac_infected and age

    combined_df["densitybin"][combined_df["densitybin"]==0] = 0.5

    age_bins = sorted(list(set(combined_df["agebin"])))
    dens_bins = sorted(list(set(combined_df["densitybin"])))
    frac_infected_bins = sorted(list(set(combined_df["fraction_infected_bin"])))
    month_bins = sorted(list(set(combined_df["month"])))

    n_age_bins = len(age_bins)
    n_frac_infected_bins = len(frac_infected_bins)
    n_subplots = n_age_bins * n_frac_infected_bins

    i = 1
    for a in age_bins:
        for f in frac_infected_bins:
            subplot_df = combined_df[np.logical_and(combined_df["agebin"]==a,
                                                    combined_df["fraction_infected_bin"]==f)]
            plt.subplot(n_age_bins,n_frac_infected_bins,i)

            c = 0
            for m, sdf in subplot_df.groupby("month"):

                plt.plot(sdf["densitybin"], sdf["reference"], label=m, c=f"C{c}", marker='o')
                plt.plot(sdf["densitybin"], sdf["simulation"], label=m, c=f"C{c}", linestyle='dashed')
                plt.xscale("log")
                c += 1
                plt.ylim([0,1])
                plt.xlim([0.5,5000])

            i+=1

    plt.tight_layout()
    #plt.show()

    #
    # plt.plot(combined_df["mean_age"], combined_df["reference"], label="reference", marker='o')
    # for param_set, sdf in combined_df.groupby("param_set"):
    #     if param_set in param_sets_to_plot:
    #         plt.plot(sdf["mean_age"], sdf["simulation"], label=f"Param set {param_set}", marker='s')
    # plt.xlabel("Age")
    # plt.ylabel("Incidence per person per year")
    # plt.title(site)
    # plt.legend()
    # plt.tight_layout()
    #plt.savefig(os.path.join(manifest.simulation_output_filepath, "_plots", f"infectious_{site}.png"))
    plt.savefig(os.path.join(plt_dir,f"infectious_{site}.png"))

def plot_infectiousness_comparison_all_sites(param_sets_to_plot=None,plt_dir=os.path.join(manifest.simulation_output_filepath, "_plots")):
    for s in infectiousness_sites:
        plot_infectiousness_comparison_single_site(s, param_sets_to_plot=param_sets_to_plot,plt_dir=plt_dir)


if __name__=="__main__":
    # plot_infectiousness_comparison_all_sites()
    #plot_infectiousness_comparison_single_site(infectiousness_sites[1])
    print(compute_infectious_LL_for_all_sites())
