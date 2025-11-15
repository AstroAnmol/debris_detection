import numpy as np
from constants import *
from collections import namedtuple

# Define a named tuple for orbital elements
OrbitalElements = namedtuple('OrbitalElements', ['a', 'e', 'i', 'Omega', 'omega', 'nu'])

# Gaussian Mixture tuple
GMMComponent = namedtuple('GMMComponent', ['weight', 'mean', 'stdDev'])

class MonteCarlo:
    """
    Generates debris orbits for Monte Carlo simulations using a combination
    of Gaussian Mixture Models (GMM) and Lognormal distributions.
    """
    def __init__(self, num_samples=int(1e+06), seed=None):
        """Initializes the random number generator."""
        if seed is not None:
            np.random.seed(seed)
        
        self.__i_GMM = [
        # Sun-Synchronous Cluster (High density, narrow spread)
        GMMComponent(weight=0.40, mean=98.0, stdDev=1.5),
        # High-Inclination Cluster (Medium density, wider spread)
        GMMComponent(weight=0.25, mean=82.0, stdDev=7.0),
        # Mid-Inclination Cluster
        GMMComponent(weight=0.25, mean=50.0, stdDev=10.0),
        # Equatorial/Low-Inclination Cluster (Small debris contribution)
        GMMComponent(weight=0.10, mean=0.0, stdDev=5.0) 
    ]
        self.__altitude_GMM = [
        # Lower LEO / Mega-constellation band
        GMMComponent(weight=0.50, mean=550.0, stdDev=75.0),
        # Higher LEO / Legacy collision band (Fengyun/Iridium)
        GMMComponent(weight=0.50, mean=950.0, stdDev=100.0)
    ]
        self.__e_lN_mu = -6.4
        self.__e_lN_std = 1.0

        self.__OEs = self.__sample_orbits(num_samples)
        
    def __sample_gmm(self, mixture, num_samples):
        """Vectorized sampling from a Gaussian Mixture Model."""
        
        # Unpack weights, means, and standard deviations
        weights = np.array([c.weight for c in mixture])
        means = np.array([c.mean for c in mixture])
        std_devs = np.array([c.stdDev for c in mixture])
        
        # 1. Select components based on weights (using a weighted choice index)
        # np.random.choice returns the index of the selected component.
        component_indices = np.random.choice(len(mixture), 
                                             size=num_samples, 
                                             p=weights)
        
        # 2. Vectorized sampling from the selected Normal distributions
        # For each sample, we use the mean and stdDev corresponding to its selected index.
        sampled_means = means[component_indices]
        sampled_std_devs = std_devs[component_indices]
        
        # Generate samples based on the appropriate mean and std dev
        samples = np.random.normal(loc=sampled_means, scale=sampled_std_devs)
        
        return samples
    
    def __convert_h_to_a(self, h_array):
        """Converts sampled altitude (h) to Semimajor Axis (a)."""
        # Semimajor axis is approximately Earth Radius + Altitude (assuming low eccentricity)
        return h_array + R_EARTH

    def __sample_orbits(self, num_samples):
        """
        Generates a specified number of debris orbital element sets.
        
        Args:
            num_samples (int): The number of orbit sets to generate.

        Returns:
            OrbitalElements: A named tuple containing numpy arrays of the sampled COEs.
        """
        
        # 1. Inclination (i) - Non-Uniform (GMM)
        i_samples = self.__sample_gmm(self.__i_GMM, num_samples)
        # Some GMM components may produce negative inclinations (e.g., -10 deg)
        # Treat negative inclinations as their supplementary angle by adding 180 deg
        # before applying the physical clipping. This maps, for example, -10 -> 170 deg.
        i_samples = np.where(i_samples < 0.0, i_samples + 180.0, i_samples)
        # Apply physical constraint: 0 <= i <= 180 degrees
        i_samples = np.clip(i_samples, I_MIN_LEO, I_MAX_LEO)
        i_samples = np.radians(i_samples)  # Convert to radians
        
        # 2. Semimajor Axis (a) - Non-Uniform (GMM, sampled via Altitude)
        h_samples = self.__sample_gmm(self.__altitude_GMM, num_samples)
        # Apply physical constraint: LEO altitude 200 km to 2000 km
        h_samples = np.clip(h_samples, H_MIN_LEO, H_MAX_LEO)
        a_samples = self.__convert_h_to_a(h_samples)

        # 3. Eccentricity (e) - Non-Uniform (Lognormal)
        e_samples = np.random.lognormal(mean=self.__e_lN_mu,
                                        sigma=self.__e_lN_std, 
                                        size=num_samples)
        
        # Apply physical constraint: 0 <= e <= E_MAX_LEO
        e_samples = np.clip(e_samples, 0.0, E_MAX_LEO)

        # 4. Angles (Ω, ω, ν) - Uniform (Randomized)
        # Right Ascension of the Ascending Node (RAAN)
        Omega_samples = np.random.uniform(ANGLE_MIN, ANGLE_MAX, size=num_samples)
        # Argument of Perigee
        omega_samples = np.random.uniform(ANGLE_MIN, ANGLE_MAX, size=num_samples)
        # True Anomaly
        nu_samples = np.random.uniform(ANGLE_MIN, ANGLE_MAX, size=num_samples)
        print(f"Successfully generated {num_samples} debris orbits.")
        
        return OrbitalElements(
            a=a_samples,
            e=e_samples,
            i=i_samples,
            Omega=Omega_samples,
            omega=omega_samples,
            nu=nu_samples
        )
    
    def get_OE_samples(self):
        """Returns the sampled orbital elements."""
        return self.__OEs
    
    def plot_OE_samples(self):
        """Plots histograms of the sampled orbital elements."""
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(3, 2, figsize=(12, 10))
        axs = axs.ravel()

        axs[0].hist(self.__OEs.a - R_EARTH, bins=50, color='blue', alpha=0.7)
        axs[0].set_title('Semimajor Axis (Altitude) Distribution')
        axs[0].set_xlabel('Altitude (km)')
        axs[0].set_ylabel('Frequency')

        axs[1].hist(self.__OEs.e, bins=50, color='green', alpha=0.7)
        axs[1].set_title('Eccentricity Distribution')
        axs[1].set_xlabel('Eccentricity')
        axs[1].set_ylabel('Frequency')

        axs[2].hist(np.degrees(self.__OEs.i), bins=50, color='red', alpha=0.7)
        axs[2].set_title('Inclination Distribution')
        axs[2].set_xlabel('Inclination (degrees)')
        axs[2].set_ylabel('Frequency')

        axs[3].hist(np.degrees(self.__OEs.Omega), bins=50, color='purple', alpha=0.7)
        axs[3].set_title('RAAN Distribution')
        axs[3].set_xlabel('RAAN (degrees)')
        axs[3].set_ylabel('Frequency')

        axs[4].hist(np.degrees(self.__OEs.omega), bins=50, color='orange', alpha=0.7)
        axs[4].set_title('Argument of Perigee Distribution')
        axs[4].set_xlabel('Argument of Perigee (degrees)')
        axs[4].set_ylabel('Frequency')

        axs[5].hist(np.degrees(self.__OEs.nu), bins=50, color='cyan', alpha=0.7)
        axs[5].set_title('True Anomaly Distribution')
        axs[5].set_xlabel('True Anomaly (degrees)')
        axs[5].set_ylabel('Frequency')

        plt.tight_layout()
        plt.show()

    def define_distributions(self, i_GMM, altitude_GMM, e_lN_mu, e_lN_std):
        self.__i_GMM = i_GMM
        self.__altitude_GMM = altitude_GMM
        self.__e_lN_mu = e_lN_mu
        self.__e_lN_std = e_lN_std