import math

# Given values
A_val = 2.533  # Minimum asymptote (corresponds to 'A' in the formula)
B_val = 1.128  # Hill's slope (corresponds to 'B' in the formula)
C_val = 94374380  # Inflection point / EC50 (corresponds to 'C' in the formula)
D_val = 47.840  # Maximum asymptote (corresponds to 'D' in the formula)
G_val = 2947221  # Asymmetry factor (corresponds to 'G' in the formula)
y_val = 2.915  # Given response value

# Calculate x using the formula:
# x = C * [ ( ((A - D) / (y - D))**(1/G) ) - 1 ]**(1/B)

# Step 1: Calculate the term ((A - D) / (y - D))
term1_numerator = A_val - D_val
term1_denominator = y_val - D_val

# Check for division by zero, though unlikely with these values for a valid y
if term1_denominator == 0:
    print("Error: Division by zero (y - D is zero). Cannot calculate x.")
    x_calculated = None
else:
    term1 = term1_numerator / term1_denominator

    # Check if term1 is negative, which would cause issues with fractional exponents
    # if G is even or if the result of term1**(1/G) - 1 is negative and B is fractional/even denominator.
    # With A < y < D (or D < y < A), (A-D)/(y-D) should be positive.
    # A_val = 2.533, D_val = 47.840. So A-D is negative.
    # y_val = 2.915. So y-D is negative.
    # Thus, (A-D)/(y-D) is positive.
    if term1 <= 0:
        print(f"Error: The term ((A - D) / (y - D)) is {term1}, which is not positive. Cannot proceed with real-valued exponentiation for (1/G).")
        x_calculated = None
    else:
        # Step 2: Raise term1 to the power of (1/G)
        # Ensure G_val is not zero to avoid division by zero for the exponent
        if G_val == 0:
            print("Error: G_val is zero, cannot compute exponent (1/G).")
            x_calculated = None
        else:
            exponent_g = 1.0 / G_val
            term2 = math.pow(term1, exponent_g)

            # Step 3: Subtract 1
            term3 = term2 - 1

            # Check if term3 is negative, which would cause issues with fractional exponents for (1/B)
            if term3 < 0: # Using < 0 strictly because 0**(positive_power) is 0.
                               # If term3 is 0, x will be 0.
                print(f"Error: The term (( (A - D) / (y - D) )**(1/G) - 1) is {term3}, which is negative. Cannot proceed with real-valued exponentiation for (1/B).")
                print("This might indicate that the y_val is outside the range defined by A, D, and G for a real x solution, or it's very close to A.")
                x_calculated = None
            else:
                # Step 4: Raise term3 to the power of (1/B)
                # Ensure B_val is not zero
                if B_val == 0:
                    print("Error: B_val is zero, cannot compute exponent (1/B).")
                    x_calculated = None
                else:
                    exponent_b = 1.0 / B_val
                    term4 = math.pow(term3, exponent_b)

                    # Step 5: Multiply by C
                    x_calculated = C_val * term4

                    # Output the results
                    print(f"Given values:")
                    print(f"  A = {A_val}")
                    print(f"  B = {B_val}")
                    print(f"  C = {C_val}")
                    print(f"  D = {D_val}")
                    print(f"  G = {G_val}")
                    print(f"  y = {y_val}")
                    print(f"Intermediate calculations:")
                    print(f"  (A - D) = {term1_numerator}")
                    print(f"  (y - D) = {term1_denominator}")
                    print(f"  ((A - D) / (y - D)) = {term1}")
                    print(f"  ((A - D) / (y - D))**(1/G) = {term2}")
                    print(f"  ((A - D) / (y - D))**(1/G) - 1 = {term3}")
                    print(f"  ( ((A - D) / (y - D))**(1/G) - 1 )**(1/B) = {term4}")
                    print(f"Calculated value of x: {x_calculated}")