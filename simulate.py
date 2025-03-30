import planet_population_simulator_FINAL
import argparse
import pandas as pd
import time

# Function definitions remain unchanged

def main():
    parser = argparse.ArgumentParser(description="Simulate transit detections given population parameters and a star catalog.")
    
    # Arguments
    parser.add_argument("--catalog", type=str, required=True, help="Path to the star catalog CSV file")
    parser.add_argument("--rcrit", type=float, default=1.5, help="Critical radius for power-law")
    parser.add_argument("--alpha_small", type=float, default=2.0, help="Power-law index 1")
    parser.add_argument("--alpha_big", type=float, default=3.0, help="Power-law index 2")
    parser.add_argument("--sigma", type=float, default=0.2, help="Standard deviation for radius sampling")
    parser.add_argument("--sigma_i", type=float, default=1.0, help="Inclination dispersion")
    parser.add_argument("--b_m", type=float, default=2.0, help="Multiplicity parameter")
    parser.add_argument("--dist", type=str, choices=["constant", "uniform", "zipfian", "poisson"], default="poisson", help="Multiplicity distribution function")
    parser.add_argument("--eta_zero", type=float, default=0.3, help="Fraction of stars with zero planets")
    parser.add_argument("--n", type=int, default=10000, help="Number of stars to process")

    args = parser.parse_args()

    # Load the star catalog
    catalog_df = pd.read_csv(args.catalog)

    # Run the transit detection simulation
    start_time = time.time()
    results, num_zeros = planet_population_simulator_FINAL.create_transit_data(
        catalog_df, args.rcrit, args.alpha_small, args.alpha_big,
        args.sigma, args.sigma_i, args.b_m, args.dist, args.eta_zero, args.n
    )
    end_time = time.time()

    # Output results
    print(f"Simulation completed in {end_time - start_time:.2f} seconds.")
    print(f"Total transits detected: {len(results)}")
    print(f"Total zero-transit systems: {num_zeros}")

if __name__ == "__main__":
    main()
