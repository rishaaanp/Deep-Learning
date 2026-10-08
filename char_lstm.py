# ============================================================
# CHARACTER-LEVEL LSTM
# NEXT CHARACTER PREDICTION
# ============================================================

import torch
import torch.nn as nn
import torch.optim as optim

# ============================================================
# 1. TRAINING DATA
# ============================================================

text = """
machine learning is a branch of artificial intelligence.
machine learning allows computers to learn from data.
machine learning algorithms identify patterns in data.
machine learning models are used for prediction and classification.
deep learning is a part of machine learning.
deep learning uses artificial neural networks.
artificial intelligence is used in many real world applications.
neural networks can learn complex patterns from data.
a neural network consists of layers of connected neurons.
recurrent neural networks are useful for sequential data.
long short term memory networks are a type of recurrent neural network.
lstm networks can remember information for a long period of time.
lstm networks are useful for sequence prediction.
character level language models predict one character at a time.
the model learns the relationship between characters.
for example, given mach, the next character is i.
machine learning is widely used in computer science.
deep learning is widely used for image and speech recognition.
natural language processing uses machine learning techniques.
data is important for training machine learning models.
learning algorithms improve their predictions during training.
"""


# ============================================================
# 2. CREATE CHARACTER VOCABULARY
# ============================================================

chars = sorted(list(set(text)))

vocab_size = len(chars)

char_to_idx = {char: idx for idx, char in enumerate(chars)}

idx_to_char = {idx: char for idx, char in enumerate(chars)}


print("=" * 60)
print("CHARACTER-LEVEL LSTM")
print("=" * 60)

print("\nVocabulary size:", vocab_size)
print("Characters:", chars)


# ============================================================
# 3. ENCODE THE TEXT
# ============================================================

encoded_text = [char_to_idx[char] for char in text]


# ============================================================
# 4. CREATE SEQUENCES
# ============================================================

# Length of each training sequence
seq_length = 20

X = []
y = []

for i in range(len(encoded_text) - seq_length):

    # Input sequence
    input_sequence = encoded_text[i : i + seq_length]

    # Target sequence
    # Shifted one character to the right
    target_sequence = encoded_text[i + 1 : i + seq_length + 1]

    X.append(input_sequence)
    y.append(target_sequence)


# Convert to tensors

X = torch.tensor(X, dtype=torch.long)

y = torch.tensor(y, dtype=torch.long)


print("\nNumber of training sequences:", len(X))
print("Input shape:", X.shape)
print("Target shape:", y.shape)


# ============================================================
# 5. DEFINE LSTM MODEL
# ============================================================


class CharLSTM(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=2):

        super(CharLSTM, self).__init__()

        # Character embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # LSTM
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2,
        )

        # Output layer
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):

        # Convert character IDs to embeddings
        x = self.embedding(x)

        # Pass through LSTM
        output, _ = self.lstm(x)

        # Generate prediction for EVERY
        # character position
        output = self.fc(output)

        return output


# ============================================================
# 6. CREATE MODEL
# ============================================================

embedding_dim = 64
hidden_dim = 128
num_layers = 2

model = CharLSTM(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    hidden_dim=hidden_dim,
    num_layers=num_layers,
)


print("\nModel architecture:")
print(model)


# ============================================================
# 7. LOSS AND OPTIMIZER
# ============================================================

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.002)


# ============================================================
# 8. TRAIN THE MODEL
# ============================================================

epochs = 300

print("\n" + "=" * 60)
print("TRAINING")
print("=" * 60)

for epoch in range(epochs):

    # Clear gradients
    optimizer.zero_grad()

    # Forward pass
    output = model(X)

    # Reshape output and target
    #
    # output:
    # [batch, sequence, vocabulary]
    #
    # becomes:
    # [batch * sequence, vocabulary]

    output = output.reshape(-1, vocab_size)

    target = y.reshape(-1)

    # Calculate loss
    loss = criterion(output, target)

    # Backpropagation
    loss.backward()

    # Update weights
    optimizer.step()

    # Print loss
    if (epoch + 1) % 25 == 0:

        print(f"Epoch [{epoch + 1}/{epochs}] " f"Loss: {loss.item():.4f}")


# ============================================================
# 9. PREDICT NEXT CHARACTER
# ============================================================


def predict_next_char(model, input_text, top_k=5):

    model.eval()

    input_text = input_text.lower()

    # Check whether characters exist
    # in the vocabulary

    for char in input_text:

        if char not in char_to_idx:

            print(f"\nError: '{char}' " f"is not present in the vocabulary.")

            return

    # If input is longer than sequence length,
    # use only the last seq_length characters

    if len(input_text) > seq_length:

        input_text = input_text[-seq_length:]

    # Convert characters to integers

    input_sequence = [char_to_idx[char] for char in input_text]

    # Convert to tensor

    input_tensor = torch.tensor([input_sequence], dtype=torch.long)

    # Prediction

    with torch.no_grad():

        output = model(input_tensor)

        # Get output from the final character
        last_output = output[:, -1, :]

        # Convert to probabilities

        probabilities = torch.softmax(last_output, dim=1)

        # Get top predictions

        top_probabilities, top_indices = torch.topk(probabilities, top_k)

    print("\nInput:", repr(input_text))

    print("\nTop predictions:")

    for probability, index in zip(top_probabilities[0], top_indices[0]):

        character = idx_to_char[index.item()]

        percentage = probability.item() * 100

        # Display spaces clearly

        display_character = character

        if character == " ":
            display_character = "[SPACE]"

        elif character == "\n":
            display_character = "[NEWLINE]"

        print(f"  {display_character:<10} " f"{percentage:.2f}%")

    # Most probable character

    predicted_index = top_indices[0][0].item()

    predicted_character = idx_to_char[predicted_index]

    print("\nPredicted next character:", repr(predicted_character))

    return predicted_character


# ============================================================
# 10. GENERATE TEXT
# ============================================================


def generate_text(model, seed_text, num_characters=100, temperature=0.7):

    model.eval()

    generated_text = seed_text.lower()

    for _ in range(num_characters):

        # Use the most recent characters

        input_text = generated_text[-seq_length:]

        # Convert to numbers

        input_sequence = [char_to_idx[char] for char in input_text]

        input_tensor = torch.tensor([input_sequence], dtype=torch.long)

        with torch.no_grad():

            output = model(input_tensor)

            # Get final time step

            output = output[:, -1, :]

            # Temperature controls randomness

            output = output / temperature

            probabilities = torch.softmax(output, dim=1)

            # Sample next character

            predicted_index = torch.multinomial(probabilities, num_samples=1).item()

        next_character = idx_to_char[predicted_index]

        generated_text += next_character

    return generated_text


# ============================================================
# 11. DYNAMIC USER INPUT
# ============================================================

print("\n" + "=" * 60)
print("DYNAMIC NEXT-CHARACTER PREDICTION")
print("=" * 60)

print("""
Enter any sequence of characters to predict the next character.

Examples:
    mach
    learn
    deep
    neural

Type 'exit' to stop.
""")


while True:

    user_input = input("\nEnter sequence: ")

    # Remove unnecessary spaces at the ends

    user_input = user_input.strip().lower()

    # Exit

    if user_input == "exit":

        print("\nProgram ended.")
        break

    # Empty input

    if user_input == "":

        print("Please enter at least one character.")

        continue

    # Check vocabulary

    invalid_characters = [char for char in user_input if char not in char_to_idx]

    if invalid_characters:

        print("\nInvalid character(s):", invalid_characters)

        print("Please use characters present " "in the training vocabulary.")

        continue

    # Predict

    predict_next_char(model, user_input)


# ============================================================
# 12. OPTIONAL TEXT GENERATION
# ============================================================

print("\n" + "=" * 60)
print("TEXT GENERATION")
print("=" * 60)

seed = input("\nEnter a starting sequence " "for text generation: ").lower()


if seed != "":

    generated = generate_text(model, seed, num_characters=100, temperature=0.7)

    print("\nGenerated text:")
    print("-" * 60)
    print(generated)
