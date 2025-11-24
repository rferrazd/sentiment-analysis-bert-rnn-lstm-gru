# GRUs


**Gated Recurrent Units (GRUs)** are a type of recurrent neural network (RNN) architecture designed to address the shortcomings of traditional RNNs, especially their difficulty in capturing long-term dependencies due to the vanishing gradient problem. GRUs introduce gating mechanisms—specifically, an update gate and a reset gate—that help the network learn when to retain past information and when to discard it. It is deciding which information it will keep and which one will not to update the hidden state. This gatting mechanism allows GRUs to remember important information over longer sequences and train more effectively, while being simpler and requiring fewer parameters than LSTMs.

## *Update Gate:* select component from previous and current hidden state

h_t = U_g*(h^_t) + (1-U_g)*h_t-1

where:
    h^_t = candidate hidden state
    (1-U_g) = update gate

- if U_g close to 0: prioritize old information
- if U_g close to 1: prioritize new information

**Update Gate Formula:**

U_g = σ(x_t·W_xu + h_{t-1}·W_hu + b_u) --> U_g = σ(W_u · [h_{t-1}, x_t] + b_u)

where:
- σ = sigmoid activation function, hence U_g [0,1]
- W_u = weight matrix for the update gate
- h_{t-1} = previous hidden state
- x_t = current input
- b_u = bias term for the update gate

**Candidate Hidden State Formula:**

ĥ_t = tanh(W_h·[R_g ⊙ h_{t-1}, x_t] + b_h)

where:  
- ĥ_t = candidate hidden state  
- R_g = reset gate
- ⊙ = element-wise multiplication  
- h_{t-1} = previous hidden state  
- x_t = current input  
- W_h = weight matrix for the candidate hidden state  
- b_h = bias for the candidate hidden state  
- tanh = hyperbolic tangent activation function  

**Reset Gate Formula:**

R_g = σ(W_r · [h_{t-1}, x_t] + b_r)

where:  
- R_g = reset gate  
- σ = sigmoid activation function, so R_g ∈ [0,1]  
- W_r = weight matrix for the reset gate  
- h_{t-1} = previous hidden state  
- x_t = current input  
- b_r = bias for the reset gate  