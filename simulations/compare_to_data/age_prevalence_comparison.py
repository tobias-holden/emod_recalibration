import os
import warnings

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
sns.set_context("talk")
sns.set_style("white")
from scipy.stats import binom

import math
from scipy.special import gammaln

from create_plots.helpers_reformat_sim_ref_dfs import get_mean_from_upper_age, \
    match_sim_ref_ages
from simulations import manifest

from simulations.load_inputs import load_sites
from simulations.helpers import load_coordinator_df

coord_csv = load_coordinator_df(characteristic=False, set_index=True)

prevalence_sites = []
sites = load_sites()
for site in sites:
    if coord_csv.at[site, 'age_prevalence'] == 1 :
        prevalence_sites.append(site)
print("Asexual Prevalence:")
print(prevalence_sites)
coord_csv = pd.read_csv(manifest.simulation_coordinator_path)

#prevalence_sites.remove('namawala_2001')

def prepare_prevalence_comparison_single_site(sim_df, site):
    """
        Read in, align, and combine reference and simulation data for a single site with an prevalence-by-age validation relationship.
        Args:
            sim_df: Simulation data from data-grabbing analyzer.
            Assumed to have the columns something like "sample_number", "Run_Number", ...

        Returns: A dataframe containing the combined reference and simulation data for this validation relationship
        """

    # Process simulation data
    #fixme sample_number column will be renamed based on upstream analyzer output
    print(site)
    sim_df.rename(columns={'PfPR': 'prevalence'}, inplace=True)

    upper_ages = sorted(sim_df['agebin'].unique())
    sim_df['mean_age'] = sim_df['agebin'].apply(get_mean_from_upper_age, upper_ages=upper_ages)
    # remove rows with population = 0 (this is relevant for cohort simulations where there is only one age group
    # in each year and all other age groups are zero)
    sim_df = sim_df[sim_df['Pop'] > 0]
        
    sim_df['month'] = sim_df['month'].astype(str)

    # Take mean over Run Number
    # sim_df = sim_df.groupby(["sample_number", "site", "Age", "p_detect_case"])\
    # sim_df = sim_df.groupby("Age").agg({"prevalence": "mean"}).reset_index()
    # Load reference data
    filepath_ref = os.path.join(manifest.base_reference_filepath,
                                coord_csv[coord_csv['site'] == site]['age_prevalence_ref'].iloc[0])
    ref_df = pd.read_csv(filepath_ref)
    ref_df = ref_df[ref_df['Site'].str.lower() == site.lower()]
    # print("REF DF")
    # print(ref_df)
    min_yr = np.min(ref_df['year'])
    print(min_yr)
    ref_df['year'] = ref_df['year']-min_yr+1
    #print("REF DF")
    #print(ref_df)
    if 'variable' in ref_df.columns:
        ref_df = ref_df[ref_df['variable']=='parasites'].reset_index()
    if 'agebin' in ref_df.columns:
        upper_ages = sorted(ref_df['agebin'].unique())
        ref_df['mean_age'] = ref_df['agebin'].apply(get_mean_from_upper_age, upper_ages=upper_ages)
    elif ('PR_LAR' in ref_df.columns) and ('PR_UAR' in ref_df.columns):
        ref_df['mean_age'] = (ref_df['PR_LAR'] + ref_df['PR_UAR']) / 2
    ref_df.rename(columns={'PR_MONTH': 'month',
                           'PR': 'prevalence',
                           'N': 'total_sampled',
                           'N_POS': 'num_pos',
                           'PR_YEAR': 'year'},
                  inplace=True)

    ref_df = ref_df[["Site", "mean_age", 'month', 'total_sampled', 'num_pos', 'prevalence', 'year']]
    # remove reference rows without prevalence values
    ref_df = ref_df[~ref_df['prevalence'].isna()]

    # determine whether reference is for a single month or averaged over multiple months - subset from simulations to match
    # check whether multiple months are listed in a character string
    if (len(ref_df['month'].unique()) == 1) and (not str(ref_df['month'].iloc[0]).isdigit()):
        # todo: need code review
        # included_months = as.numeric(unlist(strsplit(ref_df_cur$month[1], ",")))
        included_months = ref_df['month'].iloc[0].split(",")
        included_months = [str(int(x)) for x in included_months]

        sim_df = sim_df[sim_df['month'].astype(str).isin(included_months)]
        if len(included_months) > 1:
            sim_df['month'] = 'multiple'
            ref_df['month'] = 'multiple'

    elif all(ref_df['month'].astype(str).str.isnumeric()):
        included_months = ref_df['month'].unique()
        included_months = [str(int(month)) for month in included_months]
        sim_df = sim_df[sim_df['month'].astype(str).isin(included_months)]
    else:
        warnings.warn(f'The month format in the {site} reference dataset was not recognized.')

    # format reference data
    ref_df['Site'] = ref_df['Site'].str.lower()
    ref_df["site_month"] = ref_df['Site'] + '_month' + ref_df['month'].astype('str')
    #ref_df["site_month"] = ref_df['Site'] + '_year' + ref_df['year'].astype('str') + '_month' + ref_df['month'].astype('str')
    ref_df.rename(columns={"prevalence": "reference", "year": "ref_year"}, inplace=True)
    ref_df = ref_df[["reference", "mean_age", "Site", "month", "site_month", "total_sampled", "num_pos", "ref_year"]]
    
    ref_df['ref.Trials'] = ref_df['total_sampled']
    ref_df['ref.Observations'] = ref_df['num_pos']
    print("REFDF")
    print(ref_df)
    # format new simulation output
    # get simulation average across seeds
    min_yr = np.min(sim_df['year'])
    sim_df['year'] = sim_df['year']-min_yr+1
    sim_df = sim_df.groupby(["param_set", 'Site', 'mean_age', 'month','year']).agg({
        #prevalence=('prevalence', np.nanmean), 
        "prevalence":"mean", "Pop":"mean"
    }).reset_index()
    sim_df['Site'] = sim_df['Site'].str.lower()
    
    sim_df["site_month"] = sim_df['Site'] + '_month' + sim_df['month'].astype('str')
    #sim_df["site_month"] = sim_df['Site'] + '_year' + sim_df['year'].astype('str') + '_month' + sim_df['month'].astype('str')
    sim_df.rename(columns={"prevalence": "simulation"}, inplace=True)
    sim_df = sim_df[["param_set", "simulation", "mean_age", "Site", "site_month","Pop"]]
    
    sim_df['Pop'].replace(0.0,np.nan,inplace=True)
    sim_df['sim.Trials'] = sim_df['Pop']
    sim_df['sim.Observations'] = sim_df['Pop']*sim_df['simulation']
    print("SIMDF")
    print(sim_df)
    #print(sim_df)
    # check that ages match between reference and simulation. if there is a small difference (<1 year, update simulation)
    # fixme - match_sim_ref_ages is currently copied over from the validation framework and it generates an additional
    # fixme - dataframe bench_df that we don't care about right now
    def _match_sim_ref_ages_simple(df):
        # simple wrapper since we do not care about bench_df
        #fixme Code cleanup
        return match_sim_ref_ages(ref_df, df)[0]

    sim_df = sim_df.groupby("param_set")\
        .apply(_match_sim_ref_ages_simple)\
        .reset_index()\
        .drop(columns="index")

    combined_df = pd.merge(ref_df, sim_df, on=['mean_age', 'site_month', 'Site'], how='inner')
    combined_df['metric'] = 'prevalence'
    #print("Combined DF")
    #print(combined_df)

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

    #combined_df['simulation'] = combined_df['simulation'].apply(_correct_extremes)
    #print(combined_df)
    
    return combined_df


def compute_prevalence_likelihood(combined_df):
    """
    Calculate an approximate likelihood for the simulation parameters for each site. This is estimated as the product,
    across age groups, of the probability of observing the reference values if the simulation means represented the
    true population mean
    Args:
        combined_df (): A dataframe containing both the reference and matched simulation output
        sim_column (): The name of the column of combined_df to use as the simulation output
    Returns: A dataframe of loglikelihoods where each row corresponds to a site-month

    """

    # only include reference sites where the sample sizes were reported
    # combined_df = combined_df.dropna(subset=['total_sampled'])

    #fixme 230328: naively assume every observation is independent.
    # Likelihood of each observation is likelihood of seeing reference data if simulation is "reality"
    binom_ll = np.vectorize(binom.logpmf) # probability mass function of binomial distribution

    combined_df["ll"] = binom_ll(combined_df["num_pos"],
                                 combined_df["total_sampled"],
                                 combined_df["simulation"])
    #print(combined_df)                             
    return combined_df["ll"].sum()

def compute_prevalence_likelihood2(combined_df):
    df = combined_df
    ll = gammaln(df['ref.Trials'] + 1) + gammaln(df['sim.Trials'] + 2) - gammaln(df['ref.Trials'] + df['sim.Trials'] + 2) + gammaln(
        df['ref.Observations'] + df['sim.Observations'] + 1) + gammaln(
        df['ref.Trials'] - df['ref.Observations'] + df['sim.Trials'] - df['sim.Observations'] + 1) - gammaln(df['ref.Observations'] + 1) - gammaln(
        df['ref.Trials'] - df['ref.Observations'] + 1) - gammaln(df['sim.Observations'] + 1) - gammaln(df['sim.Trials'] - df['sim.Observations'] + 1)
    #print(ll)
    return ll.sum(skipna=True)#mean
    
    
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
    
def compute_prev_LL_by_site(site, numOf_param_sets):
    
    sim_df = pd.read_csv(os.path.join(manifest.simulation_output_filepath, site, "prev_inc_by_age_month.csv"))
    combined_df = prepare_prevalence_comparison_single_site(sim_df, site)
    
    
    print("COMB DF")
    print(combined_df)
    
    ll_by_param_set = combined_df.groupby("param_set",group_keys=False) \
        .apply(compute_prevalence_likelihood2) \
        .reset_index() \
        .rename(columns={0: "ll"})
    
    ll_by_param_set, missing_param_sets = identify_missing_parameter_sets(ll_by_param_set, numOf_param_sets)
  
    ll_by_param_set["site"] = site
    ll_by_param_set["metric"] = "prevalence"
    
    if len(missing_param_sets) > 0:
        print(f'Warning {site} is missing param_sets {missing_param_sets} for prevalence')
    
    return ll_by_param_set


def compute_prev_LL_for_all_sites(numOf_param_sets):
    df_by_site = []
    for s in prevalence_sites:
        df_this_site = compute_prev_LL_by_site(s,numOf_param_sets)
        df_by_site.append(df_this_site)

    return pd.concat(df_by_site)


def plot_prevalence_comparison_single_site(site, param_sets_to_plot=None,plt_dir=os.path.join(manifest.simulation_output_filepath, "_plots")):
    # Plot comparison for a specific site, given specific param_set
    sim_df = pd.read_csv(os.path.join(manifest.simulation_output_filepath, site, "prev_inc_by_age_month.csv"))
    combined_df = prepare_prevalence_comparison_single_site(sim_df, site)

    if param_sets_to_plot is None:
        param_sets_to_plot = list(set(combined_df["param_set"]))

    #todo Add error bars on data
    combined_df = combined_df.groupby(["mean_age", "param_set"])\
            .agg({"reference": "mean",
                  "simulation": "mean"})\
            .reset_index()
    plt.figure()
    plt.plot(combined_df["mean_age"], combined_df["reference"], label="reference", marker='o')
    for param_set, sdf in combined_df.groupby("param_set"):
        if param_set in param_sets_to_plot:
            plt.plot(sdf["mean_age"], sdf["simulation"], label=f"Param set {param_set}", marker='s')
    plt.xlabel("Age")
    plt.ylabel("Prevalence")
    plt.title(site)
    plt.legend()
    plt.tight_layout()
    #plt.savefig(os.path.join(manifest.simulation_output_filepath, "_plots", f"prevalence_{site}.png"))
    plt.savefig(os.path.join(plt_dir,f"prevalence_{site}.png"))

def plot_prevalence_comparison_all_sites(param_sets_to_plot=None,plt_dir=os.path.join(manifest.simulation_output_filepath, "_plots")):
    for s in prevalence_sites:
        plot_prevalence_comparison_single_site(s, param_sets_to_plot=param_sets_to_plot,plt_dir=plt_dir)


if __name__=="__main__":
    #plot_prevalence_comparison_all_sites()
    print(compute_prev_LL_for_all_sites())
