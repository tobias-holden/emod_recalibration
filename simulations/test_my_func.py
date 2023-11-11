from run_sims import submit_sim
from run_analyzers import run_analyzers
import argparse
import params as params
import manifest as manifest

if __name__ == "__main__":
    # TBD: user should be allowed to specify (override default) erad_path and input_path from command line 
    # plan = EradicationBambooBuilds.MALARIA_LINUX
    # print("Retrieving Eradication and schema.json from Bamboo...")
    # get_model_files( plan, manifest )
    # print("...done.")

    parser = argparse.ArgumentParser(description='Process site name')
    parser.add_argument('--site', '-s', type=str, help='site name',
                        default="test_site")  # params.sites[0]) # todo: not sure if we want to make this required argument
    parser.add_argument('--nSims', '-n', type=int, help='number of simulations', default=params.nSims)
    parser.add_argument('--characteristic', '-c', action='store_true', help='site-characteristic sweeps')
    parser.add_argument('--not_use_singularity', '-i', action='store_true',
                        help='not using singularity image to run in Comps')
    parser.add_argument('--priority', '-p', type=str,
                        choices=['Lowest', 'BelowNormal', 'Normal', 'AboveNormal', 'Highest'],
                        help='Comps priority', default=manifest.priority)

    args = parser.parse_args()

    submit_sim(site=args.site, nSims=args.nSims, characteristic=args.characteristic, priority=args.priority,
               not_use_singularity=args.not_use_singularity)
    run_analyzers(args.site)