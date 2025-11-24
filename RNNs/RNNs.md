# RNNs

- Common to initialize hidden sates with: h_0 = [0,0,0,0, ...], because there is no previous information
- Hidden state formula: h_t = A(x_t * W_x + h_t-1 * W_h + b_h).
  - A = activation function, could be relu, tanh, etc
  - b_h = bias vector
- Output layer formula: y_t = A'(h_t *W_y + b_y)
  - A' =. often a sigmoid of softmax

**Types of RNN Architectures:**

1. **One-to-One:**

   - Standard neural network (not an RNN).
   - Example: Image classification (one input → one output).
2. **One-to-Many:**

   - One input, produces a sequence of outputs.
   - Example: Image captioning (input an image, output a sequence of words).
   - Non sequencing input, sequencial output. Example: give  a note and then it producesa music synphony.
3. **Many-to-One:**

   - Sequence of inputs, single output.
   - It is used for problems where we have to provide the entire time series data as input, and output is based on the entire sequence.
   - No intermideate output y steps
   - Example: Sentiment analysis (input a sentence, predict a single sentiment label).
4. **Many-to-Many:**

   - Sequence of inputs, sequence of outputs.
   - Type 1 (synchronized): Part-of-speech tagging (input a sequence of words, output a sequence of tags of same length).
     - Example:
       input: 'they enjoy playing tennis'
       output: noun, verb, verb, noun
   - Type 2 (asynchronized): Machine translation (input sentence in English, output sentence in French; sequence lengths can differ).

# Advanced RNNs:


- **Multi-layer RNN**:
  - there are stacked up layers. Each layer takes as two inputs
    - input from the previous layer
    - hidden state from the previous time step.
- **Bi-directional RNN:**
  - Processes the input twice:
    - once in the forward direction and in the backward direction.
    - compute hidden states in forward and backward direction. Every time step has a concatanation of the both hidden states.
      - Ex: check image: *bidirectional_rnns_2.png*
      - Output same as before y_1 = A'(h_1_final * W_y + b_y)
    - capture long term dependencies
