import sys
import math



def main():

    if len(sys.argv) <2:
        print("Usage: python3 linear_stats.py <file_path>")
        sys.exit(1)

    file_path = sys.argv[1]

    try:
        with open(file_path, "r") as file:
            y_values = [float(line.strip()) for line in file if line.strip]
    except FileNotFoundError:
        print("File Not found")
        sys.exit(1)
    except ValueError:
        print("Error file contains invalid numerial data")
        sys.exit(1)

    n = len(y_values)

    if n == 0:
        print("File is empty")
        sys.exit(1)

    x_values = list(range(n))

    # Calculate means
    mean_x = sum(x_values) / n
    mean_y = sum(y_values) / n
    
    # Calculate components for Linear Regression and Pearson Correlation
    numerator = sum((x_values[i] - mean_x) * (y_values[i] - mean_y) for i in range(n))
    denom_x = sum((x - mean_x) ** 2 for x in x_values)
    denom_y = sum((y - mean_y) ** 2 for y in y_values)

    # y = mx + b
    # Calculate slope (m) and intercept (b) for linear regression
    slope = numerator / denom_x if denom_x != 0 else 0
    intercept = mean_y - slope * mean_x

    # Calculate Pearson correlation coefficient (r)
    r = numerator / (math.sqrt(denom_x * denom_y)) if denom_x != 0 or denom_y != 0 else 0


    print(f"Linear Regression Line: y = {slope:.6f}x + {intercept:.6f}")
    print(f"Pearson Correlation Coefficient: {r:.10f}")


if __name__ == "__main__":
    main()