import io
import torch
import imageio
import PIL

import numpy as np
import matplotlib.pyplot as plt

# https://blog.eleuther.ai/rotary-embeddings/
# https://www.3blue1brown.com/lessons/ldm-complex-numbers

# The rotation operation relies on the application of complex numbers due to their intrinsic rotational properties. 
# In the complex plane, multiplying two complex numbers results in adding their angles, 
# which effectively rotates one complex number by the angle of the other.


def figure2image():
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img = PIL.Image.open(buf)
    return img


def visualize_vectors(vectors, color="b", style="-", ax=None, fig=None):
    assert vectors.dtype == torch.complex128, f"Expected complex128, got {vectors.dtype} instead."
    
    if not ax:
        fig, ax = plt.subplots()
    elif not fig:
        fig = plt.gcf()
    
    ax.set_xlim([-0.5, 0.5])
    ax.set_ylim([-0.5, 0.5])
    for i, component in enumerate(vectors):
        ax.plot([0, component.real], [0, component.imag], color=color, linestyle=style, linewidth=1)
        ax.annotate(f"{i}", xy=(component.real, component.imag), color='r')
   
    return fig, ax


def get_cis(theta, d_model, context_length):
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

    return freqs_cis


def visualize_rope(theta, d_model, context_length, create_gif=False):
    freqs_cis = get_cis(theta, d_model, context_length)

    # take i-th token rotation angles and visualize it
    images = []
    for token_idx in range(context_length):
        plt.axis([-1.5, 1.5, -1.5, 1.5])
        rotational_embedding = freqs_cis[token_idx].squeeze().type(torch.complex128) # [1, d_model//2] 
        
        # visualize rotation component for every pair of values inside a token embedding
        for i, component in enumerate(rotational_embedding):
            plt.plot([0, component.real], [0, component.imag], color='b', linewidth=1)
            plt.annotate(f"{i}", xy=(component.real, component.imag), color='r')

        plt.title(f"{theta=}, {d_model=}, {context_length=}, {token_idx=}")
        plt.grid()
        
        image = figure2image()
        images.append(image)

        plt.close()
    
    if create_gif:
        imageio.mimsave(f"assets/{theta=}_{d_model=}_{context_length=}.gif", images)


def generate_spiral_vectors(num_vectors):
    vectors = []
    max_length = 1.0  # Maximum length for any vector component
    increment = max_length / num_vectors  # Incremental length increase

    for i in range(1, num_vectors + 1):
        length = i * increment
        angle = i * (2 * np.pi / num_vectors)  # Varying angle to create a spiral
        x = length * np.cos(angle)
        y = length * np.sin(angle)
        vectors.append([x, y])
    
    return np.array(vectors).flatten()


def visualize_embedding_rotation(theta=10, d_model=32, context_length=128, position=0, create_gif=True):
    freqs_cis = get_cis(theta, d_model, context_length)

    # mimic token embedding at {position}, coming from the model
    token_embedding = torch.Tensor(generate_spiral_vectors(d_model//2)) #torch.randn(1, d_model)
    
    # Normalize token embedding (easier to visualize the rotation effect)
    token_embedding /= torch.norm(token_embedding, dim=-1, keepdim=True)

    # reshape token embedding to pairs of values
    token_embedding = token_embedding.view(-1, 2) # [1, d_model//2]

    # convert to complex numbers to perform rotation
    token_embedding_complex = torch.view_as_complex(token_embedding.float()).type(torch.complex128)

    # visualize token embedding, splitted in pairs
    fig1, ax1 = visualize_vectors(token_embedding_complex)
    plt.grid()
    fig1.savefig(f"assets/{theta=}_{d_model=}_{context_length=}_{position=}_original.png")

    # perform rotation of pairs of values inside a token embedding
    token_embedding_rotated_complex = token_embedding_complex * freqs_cis[position]

    # visualize rotated token embedding
    fig2, ax2 = visualize_vectors(token_embedding_rotated_complex, color='g', style='--')
    plt.grid()
    fig2.savefig(f"assets/{theta=}_{d_model=}_{context_length=}_{position=}_rotated.png")

    # visualize joined original and rotated token embeddings
    fig3, ax3 = visualize_vectors(token_embedding_rotated_complex, ax=ax1, fig=fig1, color='g', style='--')
    plt.grid()
    fig3.savefig(f"assets/{theta=}_{d_model=}_{context_length=}_{position=}_joined.png")

    token_embedding_rotated = torch.view_as_real(token_embedding_rotated_complex).squeeze()


# last pairs get less rotation with higher theta
visualize_rope(theta=10000, d_model=32, context_length=128, create_gif=True)
visualize_embedding_rotation(theta=10000, d_model=32, context_length=128, position=16, create_gif=True)