import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import loguniform
import scipy.stats as stats
import mr_forecast
from astropy.io import ascii
from scipy.stats import bernoulli
import math
import torch
import time
import forecaster

start = time.time()

# P(detection|SNR)

def p_det_given_snr(snr, bl=8.06,cl=8.11,dl=0.995):
    
    return dl - dl/(1+(snr/cl)**bl)

def select_stars(catalog_df):
    return catalog_df.sample(1)

def inject_planets(distribution,b):

    # a = np.random.choice([0,1],p=[0.5,0.5])
    # if a == 1:
    #     a1 = np.random.uniform(low=b_m-minus,high=b_m)
    # elif a == 0:
    #     a1 = np.random.uniform(low=b_m,high=b_m+plus)

    if distribution == "constant":
        m = np.round(b)

    if distribution == "uniform":
        m = -1
        while m<0:
            m = np.ceil(np.random.uniform(low = b, high = 10))

    if distribution == "zipfian":
        m = 11
        while m > 10 or m < 0:
            m = np.random.zipf(1+b)
            m = np.round(m)

    if distribution == "poisson":
        m = 11
        while m > 10 or m < 1:
            m = m = np.random.poisson(b)
            m = np.round(m)
    
    return int(m)

def assign_periods(planets):
    if planets>0:
        periods = np.sort(loguniform.rvs(6.25, 400, size=planets))
    else:
        periods = np.array([])
    return np.array(periods)


# def assign_radii(planets,rcrit, alpha_small, alpha_big, sigma, rmin = 0.5, rmax = 32):

#     if planets == 0 :
    
#         return np.array([])
    
#     else:
#     # Initialize an empty list to store the sampled radii
#         radii = []
        
#         # Generate uniform random values for the log of radii
#         u = np.random.rand(1)[0]
#         # Define total normalization factor
#         norm_small = (np.log10(rcrit) - np.log10(rmin))**alpha_small
#         norm_big = (np.log10(rmax) - np.log10(rcrit))**alpha_big
#         total_norm = norm_small + norm_big
        
#         if u < norm_small / total_norm:
#             # Sample from the region [rmin, rcrit] with prob ~ (log10(r) - log10(rmin))^alpha_small
#             prob_r = np.random.uniform(np.log10(rmin), np.log10(rcrit))
#             radii.append(10**(prob_r))
#         else:
#             # Sample from the region [rcrit, rmax] with prob ~ (log10(r) - log10(rcrit))^alpha_big
#             prob_r = np.random.uniform(np.log10(rcrit), np.log10(rmax))
#             radii.append(10**(prob_r))

#         if planets>1:
#             for i in range(planets-1):
#                 r = np.abs(np.random.normal(radii[0], sigma, 1)[0])
#                 r = max(rmin, min(r, rmax)) 
#                 radii.append(r)
                

#         return np.array(radii)


def inner_rad(alpha_1,alpha_2,rcrit,rmin,rmax,size=1): 
    
    area1 = rcrit/(1-alpha_1) * (1-(rmin/rcrit)**(1-alpha_1))
    area2 = rcrit/(1-alpha_2) * ((rmax/rcrit)**(1-alpha_2)-1)
    draw = np.random.choice([0,1],1,p=[area1/(area1+area2), area2/(area1+area2)]).item()
    rand = np.random.uniform(0,1)

    if draw ==0:
        return (rand*(rcrit**(1-alpha_1)-rmin**(1-alpha_1))+rmin**(1-alpha_1))**(1/(1-alpha_1)), area1, area2
    
    elif draw==1:
        
        return (rand*(rmax**(1-alpha_2)-rcrit**(1-alpha_2))+rcrit**(1-alpha_2))**(1/(1-alpha_2)), area1, area2


def assign_radii(planets,rcrit, alpha_small, alpha_big, sigma, rmin = 0.5, rmax = 32):

    if planets == 0 :
    
        return np.array([])
    
    else:
    # Initialize an empty list to store the sampled radii
        radii = []
        radii.append(inner_rad(alpha_small,alpha_big,rcrit,rmin,rmax)[0])

        if planets>1:
            for i in range(planets-1):
                
                r = np.abs(np.random.normal(radii[0], sigma, 1)[0])
                r = max(rmin, min(r, rmax)) 
                radii.append(r)
                

        return np.array(radii)

def get_axes(periods,mstar,masses):
    G = 8.89e-10  # AU cubed per Earth mass day squared
    return (periods**2 *G*(mstar*333000+masses)/(4*np.pi**2))**(1/3)

def hill_stability(radii,mstar,periods,masses,r_crit=2*np.sqrt(3)):
    
    if len(periods) == 0 :
        return True
    
    else:

        stable = True

        if len(radii)> 1: 
            
            if masses is not None:
                axes = get_axes(periods,mstar,masses) 

                for i in range(len(radii)-1):
                    rh = ((masses[i]+masses[i+1])/(3*mstar*333000))**(1/3) * ((axes[i]+axes[i+1])/2)
                    if (axes[i+1]-axes[i])/rh <= r_crit:
                        stable = False
                        break
                
                if stable == True & len(radii) >= 3:
                    rh_in = ((masses[0]+masses[1])/(3*mstar*333000 ))**(1/3) * ((axes[0]+axes[1])/2)
                    rh_out = ((masses[-2]+masses[-1])/(3*mstar*333000 ))**(1/3) * ((axes[-2]+axes[-1])/2)

                    delta_in = (axes[1]-axes[0])/rh_in
                    delta_out = (axes[-1]-axes[-2])/rh_out
                    if delta_in + delta_out <= 18:
                        stable = False
            else:
                stable = False

        return stable

def impact_params(planets,periods,mstar,masses,rstar,sigma):

    if planets == 0 :
    
        return np.array([])
    
    else:
    
        axes = get_axes(periods,mstar,masses)
        cosi1 = np.random.uniform(low = 0, high=1)
        
        b1 = (axes[0]*23455)/(rstar*109.1) * cosi1
        
        if planets > 1:
            delta_i = np.deg2rad(np.random.rayleigh(scale=sigma, size=planets-1))
            bn_array = (axes[1:] * 23455 / (rstar * 109.1)) * np.abs(cosi1 * np.cos(delta_i) - np.sqrt(1 - cosi1**2) * np.sin(delta_i))
            bn_array = np.insert(bn_array,0,b1)
        else:
            bn_array = np.array([b1])
                    
        return np.array(bn_array)

def transit_check(impact_param,rstar,radius):
    if impact_param < (1+(radius/(rstar*109.1))):
        return True
    else:
        return False

def snr(obs_time_total,period,rp,rs,ms,mp,cdpp):
    #axes = get_axes(period,ms,mp)*23455
    #T_obs = period/np.pi*np.arcsin((np.sqrt(np.abs((rp+rs*109.1)**2-(b*rs*109.1)**2)))/axes)
    #total_t = 2*period/(2*np.pi*axes)*np.sqrt((rp+rs*109.1)**2-b**2)
    #full_t = 2*period/(2*np.pi*axes)*np.sqrt((rp-rs*109.1)**2-b**2)
    #w = (total_t+full_t)/2
    #gamma = 10**(-0.4*(kmag-12))*(1.486e10)*dutycycle
    #read_noise = 120/0.66
    #new_snr = (np.sqrt(obs_time_total/period))*w*(rp/(rs*109.1))**2*np.sqrt(gamma-read_noise)/(np.sqrt(total_t-w*(rp/(rs*109.1))**2))
    
    new_snr = (np.sqrt(obs_time_total/period))*(rp/(rs*109.1))**2/cdpp
    #print(f"new: {new_snr}")
    return new_snr

def find_closest_cdpp_duration(target_duration):
    # Parse strings to extract durations
    strings = [
    'rrmscdpp01p5', 'rrmscdpp02p0', 'rrmscdpp02p5', 'rrmscdpp03p0', 'rrmscdpp03p5',
    'rrmscdpp04p5', 'rrmscdpp05p0', 'rrmscdpp06p0', 'rrmscdpp07p5', 'rrmscdpp09p0',
    'rrmscdpp10p5', 'rrmscdpp12p0', 'rrmscdpp12p5', 'rrmscdpp15p0'
]
    durations = []
    for s in strings:
        part = s.split("rrmscdpp")[1]  # Extract the part after 'rrmscdpp'
        duration = float(part.replace('p', '.'))  # Replace 'p' with '.' and convert to float
        durations.append((s, duration))
    
    # Find the closest duration
    closest_string = min(durations, key=lambda x: abs(x[1] - target_duration))[0]
    return closest_string, float(closest_string.split("rrmscdpp")[1].replace('p', '.'))

def create_stable_planet_system(star,rcrit, alpha_small, alpha_big, sigma,b_m,dist):
    
    systems = []
    #star = select_stars(star_catalog_df)
    planet_number = inject_planets(dist,b_m)

    periods = assign_periods(planet_number)
    radii = assign_radii(planet_number,rcrit, alpha_small, alpha_big, sigma)
    masses = forecaster.Rpost2M(radii, unit='Earth')
    hill_stable = hill_stability(radii, float(star["mass"]) , periods, masses) #convert star mass from solar mass to earth mass units
    if hill_stable == True:
        for i in range(planet_number):
            systems.append([i, periods[i], radii[i],masses[i]])
    if len(systems) > 0:
        return np.array(systems)
    else: 
        return None

def create_stable_planet_system_of_transits(star,rcrit, alpha_small, alpha_big, sigma,sigma_i,b_m,dist):
    
    systems = []
    
    #star = select_stars(star_catalog_df)
    planet_number = inject_planets(dist,b_m)

    if planet_number == 0 :
        systems.append([0, 0,0,0,0])

    else:

        periods = assign_periods(planet_number)
        radii = assign_radii(planet_number,rcrit, alpha_small, alpha_big, sigma)
    
        masses = forecaster.Rpost2M(radii, unit='Earth')
        hill_stable = hill_stability(radii, float(star["mass"]) , periods, masses) #convert star mass from solar mass to earth mass units

        if hill_stable == False:
            return None

        elif hill_stable == True:
            #masses = mr_forecast.Rpost2M(radii, unit='Earth')
            bs = impact_params(planet_number,periods,float(star["mass"]),masses,float(star["radius"]),sigma_i)
            for i in range(planet_number):
                if transit_check(bs[i],float(star["radius"]),radii[i]) == True:
                    systems.append([i, periods[i], radii[i],masses[i],bs[i]])

        
    return np.array(systems)
   


def is_transit_detected(obs_time_total,period,rp,rs,ms,mp,cdpp,bl=8.06,cl=8.11,dl=0.995):
    snr_val = snr(obs_time_total,period,rp,rs,ms,mp,cdpp)*10**6
    if math.isnan(snr_val) == False:
        prob = p_det_given_snr(snr_val,bl,cl,dl)
        detected = bernoulli.rvs(prob)
    else:
        detected = 0
    return detected, snr_val


def create_transit_data(star_catalog_df, rcrit, alpha_small, alpha_big, sigma,sigma_i,b_m,dist,eta_zero,n=108014):
    
    transits = []
    sys_dicts = []
    num_trans = 0
    num_zeros = 0
    num_stars_total = 0
    

    while num_stars_total < n:
        if np.random.rand() < eta_zero:
            num_zeros = num_zeros + 1
            num_stars_total = num_stars_total + 1
        else: 
            star = select_stars(star_catalog_df)
            if star["rrmscdpp06p0"] is not np.nan:
                for iter in range(1000):
                    system_attempt = create_stable_planet_system_of_transits(star,rcrit, alpha_small, alpha_big, sigma,sigma_i,b_m,dist)
                    if system_attempt is None:
                        pass
                    elif system_attempt is not None:
                        if len(system_attempt) == 0:
                            num_zeros = num_zeros + 1
                            num_stars_total = num_stars_total + 1
                            break
                        else:
                            trans_in_sys = []
                            num_planets = len(system_attempt[:,0])
                            
                            #bs = impact_params(num_planets,system_attempt[1,:],float(star["mass"])*332943,system_attempt[3,:],float(star["radius"]),sigma_i)

                                # if int(system_attempt[-1,0]) == 0:
                                    

                                #     sys_dict = {"detected planets" : 0,
                                #                 "planet periods": np.zeros(1),
                                #                 "planet radii": np.zeros(1),
                                #                 "planet masses": np.zeros(1)}
                                #     sys_dicts.append(sys_dict)
                                #     num_trans += 1
                                #     break

                                # else:

                            for i in range(num_planets):
                            
                                b = system_attempt[i,4]
                                period = system_attempt[i,1]
                                rp = system_attempt[i,2]
                                rs = float(star["radius"])
                                ms = float(star["mass"])
                                mp = system_attempt[i,3]
                                axes = get_axes(period,ms, mp)*23455

                                t_dur = (period/np.pi*np.arcsin((np.sqrt(np.abs((rp+rs*109.1)**2-(b*rs*109.1)**2)))/axes))*24
                                
                                cdpp = float(star[find_closest_cdpp_duration(t_dur)[0]])*np.sqrt(find_closest_cdpp_duration(t_dur)[1]/t_dur)
                                detected_or_not, snr = is_transit_detected(float(star["dataspan"]),period,rp,rs,ms,mp,cdpp)
                                if detected_or_not == 1:
                                    trans_in_sys.append(system_attempt[i,:])  
                                    
                            trans_in_sys = np.array(trans_in_sys)
                                                        
                            if len(trans_in_sys) > 0:
                                num_trans += len(trans_in_sys)
                                sys_dict = {"detected planets" : len(trans_in_sys)}
                                transits.append(np.array(trans_in_sys))
                                sys_dicts.append(sys_dict)
                                num_stars_total = num_stars_total + 1
                                break
                            else:
                                num_zeros = num_zeros + 1
                                num_stars_total = num_stars_total + 1
    # for system in systems:
    #     trans_in_sys = []
    #     num_planets = len(system[0]) 
    #     for i in range(num_planets):
    #         detected_or_not = is_transit_detected(star["dataspan"],system[1][i],system[2][i],star["radius"],star["rrmscdpp06p0"],8.06, 8.11,0.995)
    #         if detected_or_not == 1:
    #             trans_in_sys.append(system[:,i])
    #     transits.append(trans_in_sys)
    print("zeros: ", num_zeros,"num_stars_total: ",num_stars_total)
    return sys_dicts, num_zeros
    



all_candidates = pd.read_csv("cumulative_2024.10.01_08.07.06.csv",delimiter=",", comment="#")
masked_cand = all_candidates.where((all_candidates['koi_period'] > 6.25)&(all_candidates['koi_period']< 400))
masked_cand = masked_cand.where((masked_cand["koi_prad"]>0.5)&(masked_cand["koi_prad"]<32))
masked_cand = masked_cand.where((masked_cand["koi_slogg"]>4))
masked_cand = masked_cand.where((masked_cand["koi_smass"]<1.2)&(masked_cand["koi_smass"]>0.8)).dropna(how='all')

# Stars
all_dr25 = pd.read_csv("result.csv")
selected_stars = all_dr25.where((all_dr25["log(g)"]>4)&(all_dr25["Mass"]<1.2)&(all_dr25["Mass"]>0.8)).dropna(how='all')
len(selected_stars)

data = ascii.read("nph-nstedAPI.txt")
all_stars = data.to_pandas()
all_stars_in_selected_stars = all_stars[all_stars['kepid'].isin(selected_stars["KIC"])]


num_param = 1
#all_params = torch.load("posterior_samples")
rcrits  = np.linspace(6,22,num_param)
alpha_smalls  = np.linspace(2.7,3.3,num_param)
alpha_bigs  = np.linspace(7,9,num_param)
sigmas  = np.linspace(0.5,0.6,num_param)
sigma_is  = np.linspace(1.1,2.7,num_param)
bs  = np.linspace(2.7,3.2,num_param)
etas = np.linspace(0.23,0.59,num_param)

data_x = torch.zeros((1,7))
# data_y = torch.zeros((num_samples,11,3))
data_y = torch.zeros((1,11))

for eta in etas:
    print("eta: ", eta)
    for rcrit in rcrits:
        for alpha_small in alpha_smalls:
            for alpha_big in alpha_bigs:
                if alpha_big < alpha_small:
                    alpha_big_new = alpha_small
                    alpha_small_new = alpha_big
                    alpha_big = alpha_big_new
                    alpha_small = alpha_small_new
                for sigma in sigmas:
                    for sigma_i in sigma_is:
                        for b in bs:
                            
                            params = torch.Tensor([rcrit,alpha_small,alpha_big,sigma,sigma_i,b,eta]).reshape((1,7))
                            data_x = torch.cat((data_x,params),0)
                            data, zeros = create_transit_data(all_stars_in_selected_stars,rcrit, alpha_small, alpha_big, sigma,sigma_i,b,dist = "poisson",eta_zero=eta)
                            
                            obs_mult = []
                            for j in data:
                                obs_mult.append(j["detected planets"])
                            hist1, _ = np.histogram(obs_mult,bins=range(1,12))
                            hist1 = np.insert(hist1,0,zeros)
                            data_y = torch.cat((data_y, torch.from_numpy(hist1).reshape((1,11))),0)
    
    
#torch.save(data_x[1:],f"data_x_params_fixstars_13")
#torch.save(data_y[1:],f"data_y_mult_hist_fixstars_13")
#torch.save(data_y,"simulation_from_sampled_posterior")


from scipy.special import factorial

mults = []
lam=7
for i in range(10000):
    mults.append(inject_planets("poisson",lam))


pdf = lam**np.array(range(0,14))*np.exp(-lam)/factorial(np.array(range(0,14),dtype=int))
plt.figure(figsize=(10,6))
plt.hist(mults,bins=range(0,14),color="lightblue",density=True,log=False)
plt.plot(range(0,14),pdf,"--",color="purple")
plt.xlabel("Number of planets")
plt.title("Poisson Distribution of Multiplicities")
plt.legend([f"Poisson distribution (beta = {lam})","Sampled data"],loc="upper right")
plt.savefig("multiplicity.png")

# Periods
sampled_periods = assign_periods(10000)  # Sample 10,000 periods
#period_bins = np.logspace(np.log10(6.25), np.log10(400), 400)
period_bins = np.linspace(6.25, 400, 400)
pdf_periods = 1/(np.log(400/6.25) * (period_bins)) 
plt.figure(figsize=(10, 6))
plt.hist(sampled_periods, bins=period_bins, density=True, alpha=0.6, color="lightblue", label="Sampled Periods")
plt.plot(period_bins, pdf_periods, color="purple", label="Loguniform PDF")
plt.loglog()
plt.xlabel("Period (days)")
plt.title("Period Loguniform Distribution")
plt.legend()
plt.show()
plt.savefig("period_distribution.png")

rs = []

a1 = -0.6
a2 = 3
rc = 8

plt.figure(figsize=(10,6))

for i in range(100000):
    rs.append(inner_rad(a1,a2,rc,0.5,32)[0])

r1 = np.linspace(0.5,rc, 1000)
r2 = np.linspace(rc, 32, 1000)

totalarea = inner_rad(a1,a2,rc,0.5,32)[1]+inner_rad(a1,a2,rc,0.5,32)[2] 

pdf1 = (r1/rc)**(-a1) / totalarea
pdf2 = (r2/rc)**(-a2) / totalarea

plt.figure(figsize=(10, 6))
plt.hist(rs, bins=np.linspace(0.5,32,50),color="lightblue",density=True,log=False,label="sampled")
plt.plot(r1, pdf1, color="purple", label="alpha1")
plt.plot(r2, pdf2, color="lightgreen", label="alpha2")
plt.vlines(rc,0,0.2,linestyles="dashed", color="black", label="rcrit")
plt.legend()
plt.loglog()

plt.title("Innermost radius distribution")
plt.xlabel("Radius (Earth radii)")
plt.savefig("innermost_radius_distribution.png")


other_radii = []
plt.figure(figsize=(10,6))
masses = []

for i in range(1):
    other_radii.append(assign_radii(2,rc,a1,a2,2))
    radii = assign_radii(1000,rc,a1,a2,2)
    plt.hist(radii,bins=np.linspace(0.5,32,50),density=True,log=False,color="lightblue")
    plt.plot(np.linspace(0.5,32,50),1/np.sqrt(2*np.pi*2**2)*np.exp(-(np.linspace(0.5,32,50)-radii[0])**2/(2*2**2)),color="purple")
    masses.append(forecaster.Rpost2M(radii, unit='Earth'))


plt.figure(figsize=(10,6))
plt.hist(other_radii,bins=np.linspace(0.5,32,50),density=True,log=False,color="lightblue")
plt.title("Radii distribution for single system")
plt.xlabel("Radius (Earth radii)")
plt.savefig("one_sys_radius_distribution.png")

plt.figure(figsize=(10,6))
plt.hist(masses,bins=np.linspace(0,30,50),density=True,log=False,color="lightblue")
plt.title("Mass distribution for single system")
plt.xlabel("Mass (Earth masses)")
plt.savefig("one_sys_mass_distribution.png")


def impact_params(planets,periods,mstar,masses,rstar,sigma):

    if planets == 0 :
    
        return np.array([])
    
    else:
        
        axes = get_axes(periods,mstar,masses)
        cosi1 = np.random.uniform(low = 0, high=1)
        
        b1 = (axes[0]*23455)/(rstar*109.1) * cosi1
        
        if planets > 1:
            delta_i = np.deg2rad(np.random.rayleigh(scale=sigma, size=planets-1))
            bn_array = (axes[1:] * 23455 / (rstar * 109.1)) * np.abs(cosi1 * np.cos(delta_i) - np.sqrt(1 - cosi1**2) * np.sin(delta_i))
            bn_array = np.insert(bn_array,0,b1)
        else:
            bn_array = np.array([b1])
                    
        return np.array(bn_array),delta_i
    
deltais = []
sigma = 2
for i in range(10000):
    deltais.append(impact_params(2,np.array([10,20]),1,1,1,sigma)[1][0])


xs = np.linspace(0,30,1000)
pdf = xs/(sigma**2) * np.exp((-xs**2)/(2*sigma**2))

plt.figure(figsize=(10,6))
plt.hist(np.rad2deg(deltais),bins=np.linspace(0,30),color= "lightblue",density=True,log=False)
plt.plot(xs,pdf,"--",color="purple")
plt.title("Inclination of 2nd planet (wrt innermost)")
plt.xlabel("Inclination (degrees)")
plt.legend([f"Rayleigh distribution (sigma = {sigma})","Sampled data"],loc="upper right")
plt.savefig("delta_i_distribution.png")


end = time.time()
print(end - start)
