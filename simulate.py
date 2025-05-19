import planet_population_simulator_FINAL
import argparse
import pandas as pd
import time
import torch
import numpy as np

# Function definitions remain unchanged

def main():
    parser = argparse.ArgumentParser(description="Simulate transit detections given population parameters and a star catalog.")
    
    # Arguments
    parser.add_argument("--catalog", type=str, default="kepler_FGK_stars.csv", help="Path to the star catalog CSV file")
    parser.add_argument("--rcrit", type=float, default=1.5, help="Critical radius for power-law")
    parser.add_argument("--alpha_small", type=float, default=2.0, help="Power-law index 1")
    parser.add_argument("--alpha_big", type=float, default=3.0, help="Power-law index 2")
    parser.add_argument("--sigma", type=float, default=0.2, help="Standard deviation for radius sampling")
    parser.add_argument("--sigma_i", type=float, default=1.0, help="Inclination dispersion")
    parser.add_argument("--b_m", type=float, default=2.0, help="Multiplicity parameter")
    parser.add_argument("--dist", type=str, choices=["constant", "uniform", "zipfian", "poisson"], default="poisson", help="Multiplicity distribution function")
    parser.add_argument("--eta_zero", type=float, default=0.3, help="Fraction of stars with zero planets")
    parser.add_argument("--n", type=int, default=108014, help="Number of stars to process")
    parser.add_argument("--o", type=str, default="sample_output", help="Output label")

    args = parser.parse_args()

    # Load the star catalog
    catalog_df = pd.read_csv(args.catalog)

    data_x = torch.zeros((1,7))
    data_y = torch.zeros((1,11))

    # Run the transit detection simulation
    start_time = time.time()

    params = torch.Tensor([args.rcrit,args.alpha_small,args.alpha_big,args.sigma,args.sigma_i,args.b_m,args.eta_zero]).reshape((1,7))
    data_x = torch.cat((data_x,params),0)


    data, num_zeros = planet_population_simulator_FINAL.create_transit_data(
        catalog_df, args.rcrit, args.alpha_small, args.alpha_big,
        args.sigma, args.sigma_i, args.b_m, args.dist, args.eta_zero, args.n
    )


    obs_mult = []
    for j in data:
        obs_mult.append(j["detected planets"])
    hist1, _ = np.histogram(obs_mult,bins=range(1,12))
    hist1 = np.insert(hist1,0,num_zeros)
    data_y = torch.cat((data_y, torch.from_numpy(hist1).reshape((1,11))),0)

    end_time = time.time()

    # Output results
    print(f"Simulation completed in {end_time - start_time:.2f} seconds.")
    print(f"Total transits detected: {len(data)}")
    print(f"Total zero-transit systems: {num_zeros}")

    torch.save(data_x[1:],f"simulations_etazero/data_x_params_{args.o}")
    torch.save(data_y[1:],f"simulations_etazero/data_y_mult_hist_{args.o}")

if __name__ == "__main__":
    main()
