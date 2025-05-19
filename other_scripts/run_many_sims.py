import itertools
import os
import subprocess
import numpy as np

#n=500
#rcrits = np.random.uniform(low=0.5,high=30,size=n)  # Example critical radii
#alpha_smalls = np.random.uniform(low=-10,high=0.99,size=n)  # Example power-law index 1
#alpha_bigs = np.random.uniform(low=1.0001,high=10,size=n)  # Example power-law index 2
#sigmas = np.random.uniform(low=0.0,high=20,size=n)  # Example standard deviations for radius sampling
#sigma_is = np.random.uniform(low=0.0,high=20,size=n)  # Example inclinations
#b_ms = np.random.uniform(low=0.0,high=20,size=n)  # Example multiplicity parameters
#eta_zeros = np.random.choice(np.linspace(0.0,0.1,100),size=(3,1))  # Example fractions of stars with zero planets
#eta_zeros = np.array([0.0]).reshape((1,1))
#eta_zero = 0.0

params = np.load("emcee_params.npy")[::9]
print(len(params))
rcrits = params[:,0]
alpha_smalls = params[:,1] 
alpha_bigs = params[:,2]
sigmas = params[:,3]
sigma_is = params[:,4]
b_ms = params[:,5]
eta_zeros = params[:,6]

# Create all combinations of parameters
#param_combinations = list(itertools.product(rcrits, alpha_smalls, alpha_bigs, sigmas, sigma_is, b_ms, eta_zeros))

def generate_sbatch(rcrit, alpha_small, alpha_big, sigma, sigma_i, b_m, eta_zero):
    sbatch_content = f"""#!/bin/bash 
#SBATCH --nodes=1                        # requests 1 compute server
#SBATCH --ntasks-per-node=1             # runs 16 tasks on each server
#SBATCH --cpus-per-task=1                # uses 1 compute core per task
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:0
#SBATCH --mem=32GB
#SBATCH --job-name=simulate
#SBATCH --output=simulate.out

module purge

singularity exec --nv \\
        --overlay /scratch/vt2189/pytorch/my_pytorch.ext3:ro \\
        /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif \\
        /bin/bash -c "source /ext3/env.sh; python3 simulate.py --rcrit {rcrit} --alpha_small {alpha_small} --alpha_big {alpha_big} --sigma {sigma} --sigma_i {sigma_i} --b_m {b_m} --eta_zero {eta_zero} --o r{rcrit}_as{alpha_small}_ab{alpha_big}_s{sigma}_si{sigma_i}_bm{b_m}_ez{eta_zero} --output_transits True"
"""
    return sbatch_content

# Loop over the combinations and create, submit, and delete .sbatch files
for i in range(len(params)):
    # Generate .sbatch file content
    sbatch_content = generate_sbatch(rcrits[i], alpha_smalls[i], alpha_bigs[i], sigmas[i], sigma_is[i], b_ms[i], eta_zeros[i])
    
    # Define the filename for the .sbatch file
    sbatch_filename = f"simulate_{rcrits[i]}_as{alpha_smalls[i]}_ab{alpha_bigs[i]}_s{sigmas[i]}_si{sigma_is[i]}_bm{b_ms[i]}_ez{eta_zeros[i]}.sbatch"
    
    # Write the sbatch content to the file
    with open(sbatch_filename, 'w') as f:
        f.write(sbatch_content)
    
    # Submit the .sbatch file using subprocess
    try:
        print(f"Submitting: {sbatch_filename}")
        subprocess.run(f"sbatch {sbatch_filename}", shell=True, check=True)
        
        # Delete the .sbatch file after submission
        os.remove(sbatch_filename)
        print(f"Deleted: {sbatch_filename}")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while submitting {sbatch_filename}: {e}")
