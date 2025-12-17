import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
LossFunction8=pd.read_csv("LossFunction_8x8.txt",names=["lossfunction"])
LossFunction12=pd.read_csv("LossFunction_12x12.txt",names=["lossfunction"])
plt.figure(figsize=(10, 6))

# First scatter plot with larger dots


# Second scatter plot with smaller dots
plt.scatter(LossFunction12.index,LossFunction12, color='red', s=5,alpha=0.5, label='12 by 12 system')
plt.scatter(LossFunction8.index,LossFunction8, color='blue', s=5,alpha=0.5, label='8 by 8 system')
# Add labels and title
plt.xlabel('Iterations')
plt.ylabel('Loss Function')

# Add a legend
plt.legend()

# Show the plot
plt.show()