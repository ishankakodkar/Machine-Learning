from sympy import symbols, diff

# --- Derivatives ---
print("--- Derivatives ---")

# Define a symbol
x = symbols('x')

# Define a function
f = x**2 + 2*x + 1
print(f"Function f(x) = {f}")

# Compute the derivative of f with respect to x
df_dx = diff(f, x)
print(f"Derivative df/dx = {df_dx}")

# Evaluate the derivative at a point (e.g., x=2)
print(f"Value of derivative at x=2: {df_dx.subs(x, 2)}")


# --- Partial Derivatives and Gradients ---
print("\n--- Partial Derivatives and Gradients ---")

# Define multiple symbols
x, y = symbols('x y')

# Define a multivariable function
g = x**2 * y + 3*y**3
print(f"Function g(x, y) = {g}")

# Compute partial derivatives
dg_dx = diff(g, x)
print(f"Partial derivative ∂g/∂x = {dg_dx}")

dg_dy = diff(g, y)
print(f"Partial derivative ∂g/∂y = {dg_dy}")

# The gradient is the vector of partial derivatives: [∂g/∂x, ∂g/∂y]
print(f"Gradient of g is [ {dg_dx}, {dg_dy} ]")

# Evaluate the gradient at a point (e.g., x=1, y=2)
grad_at_point = [dg_dx.subs({x: 1, y: 2}), dg_dy.subs({x: 1, y: 2})]
print(f"Gradient at (1, 2): {grad_at_point}")


# --- Chain Rule ---
print("\n--- Chain Rule ---")
t = symbols('t')
x_t = t**2
y_t = 2*t

# g(x(t), y(t))
g_t = g.subs({x: x_t, y: y_t})
print(f"g(t) = {g_t}")

# Derivative of g with respect to t using chain rule
# dg/dt = (∂g/∂x)*(dx/dt) + (∂g/∂y)*(dy/dt)
dx_dt = diff(x_t, t)
dy_dt = diff(y_t, t)
dg_dt_chain_rule = dg_dx.subs({x: x_t, y: y_t}) * dx_dt + dg_dy.subs({x: x_t, y: y_t}) * dy_dt
print(f"dg/dt using chain rule: {dg_dt_chain_rule.simplify()}")

# Direct derivative for verification
dg_dt_direct = diff(g_t, t)
print(f"dg/dt directly: {dg_dt_direct.simplify()}")

assert dg_dt_chain_rule.simplify() == dg_dt_direct.simplify()
print("Chain rule verification passed.")
