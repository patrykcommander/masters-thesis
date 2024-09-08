def print_metrics(metrics, phase, index):
    """
    Print all metrics for a given phase and index from the metrics dictionary.
    If phase is 'all', print metrics for all phases.

    Test metrics may have different length than train and validation. Therefore
    function prints only the last metrics from the test metrics. 

    Args:
        metrics (dict): The metrics dictionary containing data for 'train', 'validation', and 'test'.
        phase (str): The phase to retrieve metrics from ('train', 'validation', 'test', or 'all').
        index (int): The index of the metrics to retrieve.
    """
    phases = ['train', 'validation', 'test']
    
    if phase == 'all':
        # Print metrics for all phases
        for phase_name in phases:
            print(f"\nMetrics for phase '{phase_name}' at index {index if phase_name != "test" else -1}:")
            phase_data = metrics.get(phase_name, {})
            for metric_name, values in phase_data.items():
                if phase_name == "test" and len(values) > 0:
                    print(f"{metric_name.capitalize()}: {values[-1]}")
                else:
                    if index < len(values):
                        print(f"{metric_name.capitalize()}: {values[index]}")
                    else:
                        print(f"{metric_name.capitalize()}: None (Index out of range)")

    elif phase in phases:
        # Print metrics for the specified phase
        print(f"\nMetrics for phase '{phase}' at index {index if phase != "test" else -1}:")
        phase_data = metrics.get(phase, {})
        for metric_name, values in phase_data.items():
            if phase == "test" and len(values) > 0:
                print(f"{metric_name.capitalize()}: {values[-1]}")
            else:
                if index < len(values):
                    print(f"{metric_name.capitalize()}: {values[index]}")
                else:
                    print(f"{metric_name.capitalize()}: None (Index out of range)")
    
    else:
        print(f"Invalid phase: {phase}. Expected one of 'train', 'validation', 'test', or 'all'.")