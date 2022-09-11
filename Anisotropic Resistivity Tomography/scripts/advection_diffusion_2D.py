# imports
import time
import numpy as np
from tensorflow import keras
import tensorflow as tf
import math
import sys
assert sys.version_info >= (3, 5)


# TensorFlow ≥2.0 is required
assert tf.__version__ >= "2.0"

################################ Load data and initial conditions #######################
# Load data and set initial condition
nx, ny = 300, 300
T = np.zeros([nx, ny], dtype=np.float64)
gamma = 40
# initialise t:
x0 = 0
y0 = -50
x = np.zeros([1, nx], dtype=np.float64)
y = np.zeros([1, ny], dtype=np.float64)

for ii in range(nx):
    x[0][ii] = -150 + 300/nx*ii
    y[0][ii] = -150 + 300/nx*ii

# boundary excluded: range 1-299 x 1-299, I suppose we are using Dirichlet boundary condition
for i in range(1, 299):
    for j in range(1, 299):
        temp1 = -((x[0][i] - x0)**2 + (y[0][j] - y0)**2)
        temp2 = 2*gamma**2
        T[i][j] = math.exp(temp1/temp2)

input_shape = (1, nx, ny, 1)  # (1,300,300,1) as original problem size

# default data type of np.zeros is np.float64
mesh = np.zeros(input_shape, dtype=np.float64)

# generate Gaussian with a blob
for i in range(nx):
    for j in range(ny):
        mesh[0][i][j][0] = T[i][j]  # + Z1[i][j] + Z2[i][j] + Z3[i][j]*0.5

# generate Gaussian with a blob
for i in range(50):
    for j in range(50):
        mesh[0][i+225][j+125][0] = mesh[0][i+225][j+125][0] + 1

# values = tf.convert_to_tensor(mesh,dtype=np.float64)
values = mesh

################################ Initializations ####################################
start_time = time.perf_counter()

# weight matrices
w1 = ([[[[0.0],        # upwind
        [0.2],
        [0.0]],

        [[0.3],
        [-1.0],
        [0.2]],

        [[0.0],
        [0.3],
        [0.0]]]])

w2 = ([[[[0.0],        # central
        [0.15],
        [0.0]],

        [[0.25],
        [-0.8],
        [0.15]],

        [[0.0],
        [0.25],
        [0.0]]]])

# print(np.array(w1).shape) # shape (1,3,3,1)
init_kernel_1 = w1
init_kernel_2 = w2

init_bias = np.zeros((1,))  # filters - need change to exact value for bias

kernel_initializer_1 = tf.keras.initializers.constant(
    init_kernel_1)  # initializer which initialize constant tensor
kernel_initializer_2 = tf.keras.initializers.constant(init_kernel_2)

bias_initializer = tf.keras.initializers.constant(init_bias)

# CNN 2D layers: now I generate CNN filters for each subdomains
# filter 1
CNN2D_1 = keras.models.Sequential([
    keras.layers.InputLayer(input_shape=(nx, ny, 1)),
    tf.keras.layers.Conv2D(1, kernel_size=3, strides=1, padding='SAME',
                           #                                activation='relu',
                           kernel_initializer=kernel_initializer_1,
                           bias_initializer=bias_initializer),
    #         tf.keras.layers.Conv2D(1, kernel_size=3, strides=1, padding='SAME',
    #                                activation='relu',
    #                                kernel_initializer=kernel_initializer_2,
    #                                bias_initializer=bias_initializer),
])

# filter 2
CNN2D_2 = keras.models.Sequential([
    keras.layers.InputLayer(input_shape=(nx, ny, 1)),
    tf.keras.layers.Conv2D(1, kernel_size=3, strides=1, padding='SAME',
                           #                                activation='relu',
                           kernel_initializer=kernel_initializer_2,
                           bias_initializer=bias_initializer),
    #         tf.keras.layers.Conv2D(1, kernel_size=3, strides=1, padding='SAME',
    #                                activation='relu',
    #                                kernel_initializer=kernel_initializer_2,
    #                                bias_initializer=bias_initializer),
])

# here set up the hyperparameters to tune in the later training process
CNN2D_1.compile(loss="mse",
                optimizer=keras.optimizers.Nadam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999))
CNN2D_2.compile(loss="mse",
                optimizer=keras.optimizers.Nadam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999))

l1_norms = np.array([])
l2_norms = np.array([])
linf_norms = np.array([])

for t in range(1000):
    # one-step scheme with central scheme
    # a = CNN2D_2.predict(values)
    # values += a

    # two-step scheme with central scheme
    a = CNN2D_2.predict(values)
    b = (a + values)
    c = (b + values)*0.5
    d = CNN2D_2.predict(c)
    values += d

    # if t %10 == 0: # save the l1 norm and l2 norm of result per 10 timesteps
    l1_norms = np.append(l1_norms, np.linalg.norm(
        values.reshape(300, 300), ord=1)/90000)
    l2_norms = np.append(l2_norms, np.linalg.norm(
        values.reshape(300, 300), ord=2)/90000)
    linf_norms = np.append(linf_norms, np.linalg.norm(
        values.reshape(300, 300), ord=np.inf)/90000)
    # np.save("/content/serial_steps/AD_2D_serial_step_{}".format(t),values.reshape(nx, ny)) # save the resultant mesh of one time step

#  Visualization omitted
#    if t == 0:
#        plt.imshow(values[0,:,:,0], vmin=0, vmax=1.0)
#        plt.axis('off')
#        fig1_name = "paper_figure/figure_1/up_2nd_"+str(t)+".jpg"
#        plt.savefig(fig1_name, dpi=200, bbox_inches='tight')
#        plt.close()
#    elif t ==250 or t == 500 or t == 1000:
#        plt.imshow(values[0,:,:,0], vmin=0, vmax=1.0)
#        plt.axis('off')
#        fig1_name = "paper_figure/figure_1/up_2nd_"+str(t)+".jpg"
#        plt.savefig(fig1_name, dpi=200, bbox_inches='tight')
#        plt.close()


end_time = time.perf_counter()
print(
    f"[INFO] Problem solved in {end_time - start_time:0.4f} seconds using serial solution.")

# save the final result to text file
np.save("/content/output/AD_2D_serial", values.reshape(nx, ny))
np.save("/content/norms/AD_2D_serial_l1_norms", l1_norms)
np.save("/content/norms/AD_2D_serial_l2_norms", l2_norms)
np.save("/content/norms/AD_2D_serial_linf_norms", linf_norms)
