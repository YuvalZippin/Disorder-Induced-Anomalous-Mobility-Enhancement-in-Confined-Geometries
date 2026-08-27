from mpmath import polylog

# Set your parameters here
alpha = 0.3
Q0 = 0.339499

# Compute Li_{-alpha}(Q0)
polylog_val = float(polylog(-alpha, Q0))

print("Li_{-alpha}(Q0) =", polylog_val)
