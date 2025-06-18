import matplotlib.pyplot as plt

def plot_flight_trajectory(data):
    """Plot the UAV flight trajectory using latitude and longitude."""
    latitudes = [float(record['Latitude (WGS84 deg)']) for record in data]
    longitudes = [float(record['Longitude (WGS84 deg)']) for record in data]
    plt.figure(figsize=(8, 6))
    plt.plot(longitudes, latitudes, marker='o', linestyle='-', alpha=0.7)
    plt.xlabel('Longitude (deg)')
    plt.ylabel('Latitude (deg)')
    plt.title('UAV Flight Trajectory')
    plt.grid(True)
    plt.tight_layout()
    plt.show() 