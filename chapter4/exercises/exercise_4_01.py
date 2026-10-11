# Exercise 4.1
# Task: Modify Example 4.1 (single neuron) so that it performs the logical OR.
#
# Idea: the neuron sums the weighted inputs plus a bias, then applies a step function.
# With weights 1 and 1 and a bias of -0.5, the output is 0 only when both inputs are 0.

def step(x):
    return 1 if x > 0 else 0


def neuron_or(x1, x2):
    w1, w2 = 1.0, 1.0
    bias = -0.5
    return step(x1 * w1 + x2 * w2 + bias)


print("x1 x2 | OR")
for x1 in (0, 1):
    for x2 in (0, 1):
        print(f" {x1}  {x2} |  {neuron_or(x1, x2)}")

# RESULT COMMENTS:
# (fill in after running)
