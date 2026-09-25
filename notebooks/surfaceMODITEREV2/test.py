import re
import math

# Path to your file
FILE_PATH = "notebooks/doo_sabin4/doo_sabin4_tau.txt"

# Scales to analyze
TARGET_SCALES = [1, 5, 10, 15]

# Store data
scale_data = {
    scale: {
        "t": None,
        "values": []
    }
    for scale in TARGET_SCALES
}

current_scale = None

with open(FILE_PATH, "r", encoding="utf-8") as file:
    for line in file:

        # Detect scale, e.g. [ECHELLE 01]
        scale_match = re.search(r"\[ECHELLE\s+(\d+)\]", line)

        if scale_match:
            current_scale = int(scale_match.group(1))

            if current_scale not in TARGET_SCALES:
                current_scale = None

            continue

        # Read t, e.g.:
        # t         = 0.045294
        if current_scale is not None:
            t_match = re.search(
                r"^\s*t\s*=\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)",
                line
            )

            if t_match:
                scale_data[current_scale]["t"] = float(t_match.group(1))
                continue

        # Read point values, e.g.:
        # p000000  -0.003909
        value_match = re.search(
            r"p\d+\s+([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)",
            line
        )

        if value_match and current_scale is not None:
            value = float(value_match.group(1))
            # Ignore NaN and infinite values
            if math.isfinite(value):
                scale_data[current_scale]["values"].append(value)


# Print results
print("=" * 80)
print("RESULTS")
print("=" * 80)

for scale in TARGET_SCALES:

    t = scale_data[scale]["t"]
    values = scale_data[scale]["values"]

    if not values:
        print(f"\nScale {scale}: No valid values found.")
        continue

    min_value = min(values)
    max_value = max(values)

    min_abs_value = min(abs(v) for v in values)
    max_abs_value = max(abs(v) for v in values)

    print(f"\nScale {scale}")
    print(f"  t                      : {t:.6f}")
    print(f"  Number of valid values : {len(values)}")
    print(f"  Minimum value          : {min_value:+.6f}")
    print(f"  Maximum value          : {max_value:+.6f}")
    print(f"  Minimum absolute value : {min_abs_value:.6f}")
    print(f"  Maximum absolute value : {max_abs_value:.6f}")