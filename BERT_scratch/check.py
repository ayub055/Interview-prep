# Given parameters
a = 2.533
b = 1.128
c = 94374380
d = 47.840
g = 2947221
y = 2.915

def five_pl_inverse(y, A, D, C, B, G):
    """Inverse 5PL: x(y)"""
    # Avoid division by zero or invalid roots
    if y == D:
        raise ValueError("y cannot be equal to D")
    base = (A - D) / (y - D)
    if base <= 0:
        raise ValueError("Invalid y for real solution")
    inner = base ** (1/G) - 1
    if inner < 0 and B % 2 == 0:
        raise ValueError("No real solution for even B and negative inner value")
    return C * (inner) ** (1/B)

def five_pl(x, a, d, c, b, g):
    return d + (a - d) / ((1 + (x/c)**b) ** g)

# Step 1: Find x given y
x = five_pl_inverse(y, a, d, c, b, g)
# print(f"Recovered x from y={y}: {x}")

# Step 2: Plug x back into the original formula to check y
x = 8.7
y_check = five_pl(x, a, d, c, b, g)
print(f"y computed from x={x} || {y_check}")

# Optional: Check closeness
# print(f"Difference between original y and computed y: {abs(y - y_check)}")
