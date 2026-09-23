

# Python code to calculate the Statistic of a given dataset of numbers
import math

def ReadFile(data):
    with open(f"math-skills/{data}", "r") as file:
        numbers = [float(line.strip()) for line in file]

    return numbers

numbers = ReadFile("data.txt")
print("Numbers: ", numbers)


def average(numbers):
    return round(sum(numbers)/len(numbers))

mean = average(numbers)
print("Average: ", mean)

def median(numbers):
    numbers.sort()
    n = len(numbers)
    if n%2 == 0:
        median = (numbers[n//2 - 1] + numbers[n//2]) / 2
    else:
        median = numbers[n//2]
    return round(median)

print("Median: ", median(numbers))

# Variance
def variance(numbers, average, numOfData):
    squaredsum = sum([(num-average)**2 for num in numbers])
    return round(float(squaredsum/(numOfData-1)))

var = variance(numbers, mean, len(numbers))
print("Variance: ", var)


def standardDeviation(variance):
     return round(math.sqrt(variance))

StandDiv = standardDeviation(var)
print("Standard Deviation: ", StandDiv)




