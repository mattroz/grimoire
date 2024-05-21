import torch

import matplotlib.pyplot as plt

# https://blog.eleuther.ai/rotary-embeddings/
# https://www.3blue1brown.com/lessons/ldm-complex-numbers

# The rotation operation relies on the application of complex numbers due to their intrinsic rotational properties. 
# In the complex plane, multiplying two complex numbers results in adding their angles, 
# which effectively rotates one complex number by the angle of the other.


def visualize_rope(theta, d_model, context_length):
    thetas = 1. / theta ** (2 * torch.arange(d_model//2) / d_model) # [d_model // 2] - rotation angles for every pair of tokens.

    # thetas_per_token matrix visualization, m - position in a sequence, θ - rotation angle.
    #___________________________________________________________________
    # m = 1 | 1 * θ1 |  1 * θ2 |  1 * θ3  |  ...  |  1 * θ_{d_model//2} |
    # m = 2 | 2 * θ1 |  2 * θ2 |  2 * θ3  |  ...  |  2 * θ_{d_model//2} |
    # m = 3 | 3 * θ1 |  3 * θ2 |  3 * θ3  |  ...  |  3 * θ_{d_model//2} |
    # m = n | .......|.........|......... |...... |.................... |
    # m = T | T * θ1 |  T * θ2 |  T * θ3  |  ...  |  T * θ_{d_model//2} |
    #___________________________________________________________________|
    thetas_per_token = torch.outer(torch.arange(context_length), thetas) # [context_length, d_model//2] - every token embedding has its own rotation angles.

    # In math terms, we can represent a complex number as a + bi, where a is the real part and b is the imaginary part.
    # If you have a vector (v1, v2) and you want to rotate this vector by `theta` degrees, 
    # you can multiply it by a complex number `cis(theta) = cos(theta) + i * sin(theta)`:
    # (v1,v2)_rotated = (v1 + i * v2) * cis(theta).

    # So, the whole point of `freqs_cis` tensor (naming comes from LLaMA, IIRC) is to store the CIS for every pair of values inside a token embedding.
    # We use torch.ones_like, because we need to perform rotation only without changing the vector length.
    freqs_cis = torch.polar(torch.ones_like(thetas_per_token), thetas_per_token) # [context_length, d_model//2]

    # take i-th token rotation angles and visualize it
    for token_idx in range(context_length):
        plt.axis([-1.5, 1.5, -1.5, 1.5])
        rotational_embedding = freqs_cis[token_idx].squeeze().type(torch.complex128) # [1, d_model//2] 
        
        # visualize rotation component for every pair of values inside a token embedding
        for i, component in enumerate(rotational_embedding):
            plt.plot([0, component.real], [0, component.imag], color='b', linewidth=1)
            plt.annotate(f"{i}", xy=(component.real, component.imag), color='r')

        plt.title(f"{theta=}, {d_model=}, {context_length=}, {token_idx=}")
        plt.grid()
        plt.savefig(f"images/rotational_embedding_{token_idx}.png")
        plt.close()

# last pairs get less rotation with higher theta
visualize_rope(theta=10, d_model=32, context_length=256)
