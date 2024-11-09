import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set the sample size
n = 1000

# Set the blood pressure estimate for method A and method B
theta_hat_a = np.random.normal(120, 10, n)  # Method A
theta_tilda_b = np.random.normal(120, 10, n)  # Method B

# Applying the transformation to the blood pressure estimate to log scale
gamma_hat_a = np.log(theta_hat_a)  # Basically the log of the blood pressure estimate for method A
gamma_tilda_b = np.log(theta_tilda_b)  # Basically the log of the blood pressure estimate for method B

# Calculate the variance of the blood pressure estimate for method A and method B
var_gamma_hat_a = np.var(gamma_hat_a)
var_gamma_tilda_b = np.var(gamma_tilda_b)

# Calculate the efficient avar for method A and method B
avar_gamma_hat_a = var_gamma_hat_a / n
avar_gamma_tilda_b = var_gamma_tilda_b / n

# Compare the efficient avar for method A and method B
if avar_gamma_tilda_b > avar_gamma_hat_a:
    print('Method A is more efficient than Method B')
else:
    print('Method B is more efficient than Method A')

# Display the distribution of the blood pressure estimate for method A and method B
sns.kdeplot(gamma_hat_a, label='Method A', common_norm=False)
sns.kdeplot(gamma_tilda_b, label='Method B', common_norm=False)
plt.legend()
plt.title('Distribution of the blood pressure estimate for Method A and Method B')
plt.xlabel('Blood Pressure Estimate')
plt.ylabel('Density')
plt.show()

