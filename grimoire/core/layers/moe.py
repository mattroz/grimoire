import torch
import torch.nn.functional as F

from typing import List
from torch import nn


class MixtureOfExpertsLayer(nn.Module):
    """
    Mixtral of Experts [https://arxiv.org/abs/2401.04088], section 2.1
    """
    def __init__(self, experts: List[nn.Module], router: nn.Module, num_experts_per_token: int, noise_eps: float = 0.01):
        super().__init__()

        assert len(experts) > 0
        self.experts = nn.ModuleList(experts)
        self.router = router
        self.num_experts_per_token = num_experts_per_token
        self.noise_eps = noise_eps

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = inputs.shape

        # From Switch Transformers [https://arxiv.org/abs/2101.03961], section C:
        # usually while training you also want to add noise to the logits,  
        # which enables exploration so that the model doesn't always rely on the same experts)
        if self.training:
            # torch.empty_like is just faster
            inputs *= torch.empty_like(inputs).uniform_(1 - self.noise_eps, 1 + self.noise_eps)

        inputs = inputs.view(-1, hidden_dim)
        
        # [batch_size * sequence_length, num_experts]
        router_logits = self.router(inputs) 

        # router_logits: [batch_size * sequence_length, num_experts_per_token]
        # selected_experts: [batch_size * sequence_length, num_experts_per_token]
        router_logits, selected_experts = torch.topk(
            router_logits, self.num_experts_per_token
        )

        expert_importance = F.softmax(router_logits, dim=1, dtype=torch.float).to(inputs.dtype)
        final_state = torch.zeros_like(inputs) # [batch_size * sequence_length, hidden_dim]

        for i, expert in enumerate(self.experts):
            token_idx, nth_expert = torch.where(selected_experts == i)
            current_state = expert_importance[token_idx, nth_expert, None] * expert(inputs[token_idx])
            final_state[token_idx] += current_state

        return final_state.reshape(batch_size, sequence_length, hidden_dim)