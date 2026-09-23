#!/usr/bin/env python3
import sys
import math


def main():
    history = []

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            val = float(line)
        except ValueError:
            continue
        history.append(val)
        n = len(history)

        if n == 1:
            # Fallback range for the very first point
            lower = int(val - 50)
            upper = int(val + 50)
        else:
            # Using linear regression over a rolling window to capture the trend
            # A window of size 40 or 50 works best for catching trends without lagging too much
            window_size = min(n, 40)
            window_y = history[-window_size:]
            window_x = list(range(n - len(window_y), n))
            
            w_len = len(window_y)
            mean_x = sum(window_x) / w_len
            mean_y = sum(window_y) / w_len

            numerator = sum((window_x[i] - mean_x) * (window_y[i] - mean_y) for i in range(w_len))
            denom_x = sum((x - mean_x) ** 2 for x in window_x)

            # Calculate slope (m) and intercept (b)
            m = 0 if denom_x  ==0 else numerator / denom_x 
            b = mean_y - (m * mean_x)

            # Predict the value for the *next* x index (which is current n)
            next_x = n
            predicted_y = m * next_x + b

            # Calculate standard deviation of residuals/errors for sizing the range
            variance = sum((window_y[i] - (m * window_x[i] + b)) ** 2 for i in range(w_len)) / max(1, w_len - 2)
            std_dev = math.sqrt(variance)

            # Set margin: tighter if data fits the line well, wider as a safety buffer
            margin = max(std_dev * 2.2, 18.0)

            lower = int(round(predicted_y - margin))
            upper = int(round(predicted_y + margin))

            # Ensure proper ordering
            if lower > upper:
                lower, upper = upper, lower

        print(f"{lower} {upper}")


if __name__ == "__main__":
    main()