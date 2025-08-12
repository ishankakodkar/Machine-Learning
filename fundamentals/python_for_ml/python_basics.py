# Python Basics for Machine Learning

# --- Data Structures ---
print("--- Data Structures ---")

# Lists (mutable)
my_list = [1, 'hello', 3.14]
print(f"List: {my_list}")
my_list.append(4)
print(f"List after append: {my_list}")
print(f"First element: {my_list[0]}")

# Dictionaries (key-value pairs)
my_dict = {'name': 'Alice', 'age': 30, 'city': 'New York'}
print(f"\nDictionary: {my_dict}")
print(f"Age: {my_dict['age']}")
my_dict['email'] = 'alice@example.com'
print(f"Dictionary after adding email: {my_dict}")

# Tuples (immutable)
my_tuple = (1, 2, 3)
print(f"\nTuple: {my_tuple}")
# my_tuple[0] = 5 # This would raise a TypeError

# Sets (unordered, unique elements)
my_set = {1, 2, 2, 3, 4, 4, 4}
print(f"\nSet: {my_set}")


# --- Loops ---
print("\n--- Loops ---")

# For loop
print("For loop over a list:")
for item in my_list:
    print(item)

# For loop with range
print("\nFor loop with range:")
for i in range(5):
    print(i)


# --- List Comprehensions ---
print("\n--- List Comprehensions ---")

squares = [x**2 for x in range(10)]
print(f"Squares of numbers from 0 to 9: {squares}")

even_squares = [x**2 for x in range(10) if x % 2 == 0]
print(f"Squares of even numbers from 0 to 9: {even_squares}")


# --- Functions ---
print("\n--- Functions ---")

def greet(name):
    """This function greets the person passed in as a parameter."""
    return f"Hello, {name}!"

greeting = greet('Bob')
print(greeting)

def add(a, b=5):
    """This function adds two numbers. b has a default value."""
    return a + b

print(f"add(3, 2) = {add(3, 2)}")
print(f"add(3) = {add(3)}")
