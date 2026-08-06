import tensorflow as tf
from tensorflow.keras import layers, Sequential
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# -------------------------------
# 1. Load and Preprocess Dataset
# -------------------------------

(x_train, _), (_, _) = tf.keras.datasets.mnist.load_data()

# Normalize images to [-1, 1]
x_train = (x_train.astype("float32") - 127.5) / 127.5

# Reshape to (28,28,1)
x_train = np.expand_dims(x_train, axis=-1)

BUFFER_SIZE = x_train.shape[0]
BATCH_SIZE = 256
NOISE_DIM = 100
EPOCHS = 200  # Change if required

dataset = (
    tf.data.Dataset.from_tensor_slices(x_train)
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE, drop_remainder=True)
)

# -------------------------------
# 2. Generator Network
# -------------------------------


def build_generator():

    model = Sequential(
        [
            tf.keras.Input(shape=(NOISE_DIM,)),
            layers.Dense(7 * 7 * 256, use_bias=False),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            layers.Reshape((7, 7, 256)),
            layers.Conv2DTranspose(
                128, kernel_size=5, strides=1, padding="same", use_bias=False
            ),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            layers.Conv2DTranspose(
                64, kernel_size=5, strides=2, padding="same", use_bias=False
            ),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            layers.Conv2DTranspose(
                1,
                kernel_size=5,
                strides=2,
                padding="same",
                use_bias=False,
                activation="tanh",
            ),
        ]
    )

    return model


generator = build_generator()

# -------------------------------
# 3. Discriminator Network
# -------------------------------


def build_discriminator():

    model = Sequential(
        [
            tf.keras.Input(shape=(28, 28, 1)),
            layers.Conv2D(64, kernel_size=5, strides=2, padding="same"),
            layers.LeakyReLU(),
            layers.Dropout(0.3),
            layers.Conv2D(128, kernel_size=5, strides=2, padding="same"),
            layers.LeakyReLU(),
            layers.Dropout(0.3),
            layers.Flatten(),
            layers.Dense(1),
        ]
    )

    return model


discriminator = build_discriminator()

# -------------------------------
# 4. Loss Functions
# -------------------------------

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


def discriminator_loss(real_output, fake_output):

    real_loss = cross_entropy(tf.ones_like(real_output), real_output)

    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)

    return real_loss + fake_loss


# -------------------------------
# 5. Optimizers
# -------------------------------

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# -------------------------------
# 6. Training Step
# -------------------------------


@tf.function
def train_step(images):

    noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    generator_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)

    discriminator_gradients = disc_tape.gradient(
        disc_loss, discriminator.trainable_variables
    )

    generator_optimizer.apply_gradients(
        zip(generator_gradients, generator.trainable_variables)
    )

    discriminator_optimizer.apply_gradients(
        zip(discriminator_gradients, discriminator.trainable_variables)
    )

    return gen_loss, disc_loss


# -------------------------------
# 7. Train GAN
# -------------------------------

generator_losses = []
discriminator_losses = []

print("\nStarting GAN Training...\n")

epoch_progress = tqdm(range(EPOCHS), desc="Training Progress", unit="epoch")

for epoch in epoch_progress:

    g_loss_epoch = 0
    d_loss_epoch = 0
    batches = 0

    for image_batch in dataset:

        g_loss, d_loss = train_step(image_batch)

        g_loss_epoch += g_loss.numpy()
        d_loss_epoch += d_loss.numpy()

        batches += 1

    g_loss_epoch /= batches
    d_loss_epoch /= batches

    generator_losses.append(g_loss_epoch)
    discriminator_losses.append(d_loss_epoch)

    epoch_progress.set_postfix(
        G_Loss=f"{g_loss_epoch:.4f}", D_Loss=f"{d_loss_epoch:.4f}"
    )

    tqdm.write(
        f"Epoch {epoch+1:3d}/{EPOCHS} | "
        f"Generator Loss = {g_loss_epoch:.4f} | "
        f"Discriminator Loss = {d_loss_epoch:.4f}"
    )

print("\nTraining Complete!\n")

# -------------------------------
# 8. Generate 16 Images
# -------------------------------

noise = tf.random.normal([16, NOISE_DIM])

generated_images = generator(noise, training=False)

generated_images = (generated_images + 1) / 2

plt.figure(figsize=(6, 6))

for i in range(16):

    plt.subplot(4, 4, i + 1)
    plt.imshow(generated_images[i, :, :, 0], cmap="gray")
    plt.axis("off")

plt.suptitle("Generated Handwritten Digits")
plt.tight_layout()
plt.show()

# -------------------------------
# 9. Plot Generator Loss
# -------------------------------

plt.figure(figsize=(8, 5))
plt.plot(generator_losses, linewidth=2)
plt.title("Generator Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()

# -------------------------------
# 10. Plot Discriminator Loss
# -------------------------------

plt.figure(figsize=(8, 5))
plt.plot(discriminator_losses, linewidth=2)
plt.title("Discriminator Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()

# -------------------------------
# 11. Final Loss Values
# -------------------------------

print("=" * 50)
print("Training Summary")
print("=" * 50)

print(f"Final Generator Loss     : {generator_losses[-1]:.4f}")
print(f"Final Discriminator Loss : {discriminator_losses[-1]:.4f}")

print("=" * 50)
