import argparse
from data_loader import load_flight_data
from visualization import plot_flight_trajectory
from metrics import compute_flight_metrics


def main():
    parser = argparse.ArgumentParser(description='FLY-Bench UAV Flight Data Analysis')
    parser.add_argument('--data', type=str, required=True, help='Path to the flight data JSONL file')
    args = parser.parse_args()

    # Load flight data
    data = load_flight_data(args.data)

    # Basic analysis: print summary
    print(f"Loaded {len(data)} flight records.")
    print("First record:")
    print(data[0])

    # Compute and print metrics
    metrics = compute_flight_metrics(data)
    print("Flight Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    # Visualize flight trajectory
    plot_flight_trajectory(data)

if __name__ == '__main__':
    main() 