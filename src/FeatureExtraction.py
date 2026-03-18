"""
FeatureExtraction - Filtering the iris and extracting features.
"""


def extract_features():
    pass


def circularly_symmetric_filter(size, sigma_x, sigma_y, f):

    """
    Generates a circularly symmetric spatial filter as defined in Equation 3.
    The kernel uses a circularly symmetric modulating sinusoidal function.
    """

    # Define an X-Y grid
    half_size = size // 2
    x = np.arange(-half_size, half_size + 1)
    y = np.arange(-half_size, half_size + 1)
    X, Y = np.meshgrid(x, y)

    # Gaussian envelope
    envelope = (1 / (2 * np.pi * sigma_x * sigma_y)) * np.exp(-0.5 * ((X**2 / sigma_x**2) + (Y**2 / sigma_y**2)))
 
    # Circularly symmetric sinusoidal function
    modulator1 = np.cos(2 * np.pi * f * np.sqrt(X**2 + Y**2))

    # Oriented sinusoidal function, maybe needed in the future for comparison
    # modulator2 = np.cos(2 * np.pi * f  * (X*np.cos(theta) + Y*np.sin(theta)))

    # Gabor filter
    Gabor_filter = envelope * modulator1

    return Gabor_filter

if __name__ == "__main__":
    extract_features()
