import json

def load_flight_data(filepath):
    """Load flight data from a JSONL file."""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            record = json.loads(line)
            data.append(record)
    return data 