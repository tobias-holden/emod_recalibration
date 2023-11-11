import sys
sys.path.append('../')
import pandas as pd
import os
import manifest as manifest

from simulations.compare_to_data.age_incidence_comparison import compute_inc_LL_for_all_sites, \
    plot_incidence_comparison_all_sites
from simulations.compare_to_data.age_prevalence_comparison import compute_prev_LL_for_all_sites, \
    plot_prevalence_comparison_all_sites
from simulations.compare_to_data.infectiousness_comparison import compute_infectious_LL_for_all_sites, \
    plot_infectiousness_comparison_all_sites
from simulations.compare_to_data.parasite_density_comparison import compute_parasite_density_LL_for_all_sites, \
    plot_density_comparison_all_sites
from simulations.compare_to_data.age_gamotocyte_prevalence_comparison import compute_gametocyte_prev_LL_for_all_sites, \
    plot_gametocyte_prevalence_comparison_all_sites

from simulations.compare_to_data.no_blood_comparison import compute_dead_LL_for_all_sites

def compute_LL_across_all_sites_and_metrics(numOf_param_sets = 64):
    #infectious_LL = compute_infectious_LL_for_all_sites(numOf_param_sets)
    #density_LL = compute_parasite_density_LL_for_all_sites(numOf_param_sets)
    prevalence_LL = compute_prev_LL_for_all_sites(numOf_param_sets)
    #gametocyte_prevalence_LL = compute_gametocyte_prev_LL_for_all_sites(numOf_param_sets)
    #incidence_LL = compute_inc_LL_for_all_sites(numOf_param_sets)
    #dead_LL = compute_dead_LL_for_all_sites(numOf_param_sets)


    #density_LL_w=density_LL
    #density_LL_w['ll'] = [float(val)/10 for val in density_LL['ll']]
    #print(density_LL_w)
    #combined = pd.concat([infectious_LL, density_LL, prevalence_LL, incidence_LL, dead_LL, gametocyte_prevalence_LL])
    combined = pd.concat([prevalence_LL])
    print(combined.to_string())

    #fixme - Eventually, this will need to be fancier way of weighting LL across the diverse metrics/sites
    #fixme - For now, just naively add all log-likelihoods
    return combined.groupby("param_set").agg({"ll": lambda x: x.sum(skipna=False)}).reset_index()

def plot_all_comparisons(param_sets_to_plot=None,plt_dir=os.path.join(manifest.simulation_output_filepath, "_plots")):
    plot_incidence_comparison_all_sites(param_sets_to_plot=param_sets_to_plot,plt_dir=plt_dir)
    plot_prevalence_comparison_all_sites(param_sets_to_plot=param_sets_to_plot,plt_dir=plt_dir)
    plot_gametocyte_prevalence_comparison_all_sites(param_sets_to_plot=param_sets_to_plot,plt_dir=plt_dir)
    plot_density_comparison_all_sites(param_sets_to_plot=param_sets_to_plot,plt_dir=plt_dir)
    plot_infectiousness_comparison_all_sites(param_sets_to_plot=param_sets_to_plot) 

if __name__ == "__main__":
    #plot_all_comparisons(param_sets_to_plot=[1.0,4.0,6.0,9.0,15.0])
    
    #if you are running directly from run_full_comparison you are going to probably want to 
    #manually add a different default numOf_param_sets, for example, numOf_param_sets = 16
    print(compute_LL_across_all_sites_and_metrics(numOf_param_sets=5))
