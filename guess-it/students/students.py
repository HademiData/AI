#!/usr/bin/env python3

import math
import sys

def variance(numbers, average, numOfData):
    squaredsum = sum([(num-average)**2 for num in numbers])
    return round(float(squaredsum/(numOfData-1)))

def average(numbers):
    return round(sum(numbers)/len(numbers))

def standardDeviation(variance):
     return round(math.sqrt(variance))


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
            lower = int(val - 70)
            upper = int(val + 70)

        else:
            window = history[-30:] if n >30 else history

            mean = average(window)
            var = variance(window, mean, len(history))
            std = standardDeviation(var)

            margin = max(std * 1.8, 15)

            lower = int(round(mean - margin))
            upper = int(round(mean + margin))

            if lower > upper:
                lower, upper = upper, lower

        print(f"{lower} {upper}")

if __name__ == "__main__":
    main()

