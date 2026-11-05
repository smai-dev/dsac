"""Create reproducible, fictional machine-condition signals for Spark experiments.

The values emulate broad equipment telemetry patterns; they are not calibrated
measurements from field equipment and must not guide real maintenance decisions.
"""
from pathlib import Path
import numpy as np
import pandas as pd


def generate_machine_telemetry(machine_count=250, days_per_machine=30, seed=2026):
    """Write daily operating signals and machine-level service labels.

    The three sensor series have different distributions from the earlier dataset.
    A shared daily workload disturbance introduces correlated variation, while a
    latent wear state produces gradual trends. The target is sampled at machine
    level, not copied from any particular reading.
    """
    if machine_count < 1 or days_per_machine < 1:
        raise ValueError("machine_count and days_per_machine must be positive")

    rng = np.random.default_rng(seed)
    wear_categories = np.array(["stable", "moderate wear", "advanced wear"])
    operating_environments = np.array([
        "Open field", "Rolling terrain", "Dry ground", "Wet ground"
    ])
    environment_load = {
        "Open field": 0.0,
        "Rolling terrain": 0.8,
        "Dry ground": 0.5,
        "Wet ground": 0.6,
    }

    signal_rows = []
    label_rows = []

    for number in range(machine_count):
        machine_id = f"EQ-{number:04d}"
        wear_state = int(rng.choice(3, p=[0.58, 0.28, 0.14]))
        environment = str(rng.choice(operating_environments, p=[0.42, 0.23, 0.20, 0.15]))
        terrain_load = environment_load[environment]

        # Per-machine offsets avoid perfectly separated wear groups.
        vibration_baseline = 2.6 + 1.65 * wear_state + rng.normal(0, 0.48)
        temperature_baseline = 79 + 8.0 * wear_state + 2.0 * terrain_load + rng.normal(0, 2.4)
        pressure_baseline = 191 - 16.0 * wear_state - 4.0 * terrain_load + rng.normal(0, 7.0)

        # Generate one binary service signal per machine using its latent condition.
        service_log_odds = -3.0 + 1.7 * wear_state + 0.3 * terrain_load + rng.normal(0, 0.58)
        service_probability = 1 / (1 + np.exp(-service_log_odds))
        maintenance_required = int(rng.random() < service_probability)

        workload = rng.normal(0, 1, size=days_per_machine)
        time_index = np.arange(days_per_machine)

        # Day-to-day variation is shared across vibration, heat, and pressure.
        vibration = (
            vibration_baseline + (0.006 + 0.019 * wear_state) * time_index
            + 0.22 * workload + rng.normal(0, 0.35, size=days_per_machine)
        )
        temperature = (
            temperature_baseline + (0.025 + 0.10 * wear_state) * time_index
            + 1.5 * workload + rng.normal(0, 2.5, size=days_per_machine)
        )
        pressure = (
            pressure_baseline - (0.07 + 0.24 * wear_state) * time_index
            - 2.4 * workload + rng.normal(0, 5.5, size=days_per_machine)
        )

        # A nonzero code flags a reading excluded by the notebook's quality rule.
        event_probabilities = [
            0.95 - 0.01 * wear_state,
            0.04 + 0.008 * wear_state,
            0.01 + 0.002 * wear_state,
        ]
        event_codes = rng.choice([0, 1, 2], days_per_machine, p=event_probabilities)

        for offset in range(days_per_machine):
            signal_rows.append({
                "machine_id": machine_id,
                "operating_day": offset + 1,
                "motor_vibration_rms": round(float(max(0.2, vibration[offset])), 2),
                "engine_temperature_c": round(float(np.clip(temperature[offset], 45, 125)), 2),
                "hydraulic_pressure_bar": round(float(np.clip(pressure[offset], 80, 235)), 2),
                "diagnostic_event_code": int(event_codes[offset]),
            })

        label_rows.append({
            "machine_id": machine_id,
            "operating_environment": environment,
            "maintenance_required": maintenance_required,
        })

    data_dir = Path(__file__).resolve().parent
    signals_file = data_dir / "machine_operating_signals.csv"
    labels_file = data_dir / "machine_maintenance_targets.csv"
    pd.DataFrame(signal_rows).to_csv(signals_file, index=False)
    pd.DataFrame(label_rows).to_csv(labels_file, index=False)
    print(f"Generated {len(signal_rows):,} daily signals -> {signals_file.name}")
    print(f"Generated {len(label_rows):,} service targets -> {labels_file.name}")
    return signals_file, labels_file


if __name__ == "__main__":
    generate_machine_telemetry()
