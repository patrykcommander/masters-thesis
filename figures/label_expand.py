import numpy as np
import matplotlib.pyplot as plt

original_vector = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])

expansion_radius = 3

expanded_vector = original_vector.copy()
for i in range(len(original_vector)):
    if original_vector[i] == 1:
        start = max(0, i - expansion_radius)
        end = min(len(original_vector), i + expansion_radius + 1)
        expanded_vector[start:end] = 1

# Step 4: Set custom x-ticks
negative_ticks = ['n - ' + str(abs(x)) for x in range(-6,0)]
positive_ticks = ['n + ' + str(x) for x in range(1,7)]
x_ticks = ['n']

x_ticks = negative_ticks + x_ticks + positive_ticks

# Step 5: Plot the original and expanded vectors
plt.figure(figsize=(8, 6))

# Plot original vector
plt.subplot(2, 1, 1)
plt.stem(range(len(original_vector)), original_vector, linefmt='b-', markerfmt='bo', basefmt='r-')
plt.title('Oryginalny wektor')
plt.ylim(-0.1, 1.1)
plt.xticks(range(len(original_vector)), x_ticks)
plt.yticks([0,1], [0,1])
plt.xlabel('Indeks')
plt.ylabel('Prawdopowodbieństwo')
plt.grid()

# Plot expanded vector
plt.subplot(2, 1, 2)
plt.stem(range(len(expanded_vector)), expanded_vector, linefmt='g-', markerfmt='go', basefmt='r-')
plt.title('Wektor po procedurze "Label Expanding"')
plt.ylim(-0.1, 1.1)
plt.xticks(range(len(expanded_vector)), x_ticks)
plt.yticks([0,1], [0,1])
plt.xlabel('Indeks')
plt.ylabel('Prawdopowodbieństwo')
plt.grid()

plt.tight_layout()
plt.show()