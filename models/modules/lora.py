import torch
from torch import nn


def replace_linear_with_lora(
    module: nn.Module,
    max_rank: int,
    scale: float = 1.0,
) -> None:
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            new_lora = LinearLora(
                in_features=child.in_features,
                out_features=child.out_features,
                bias=child.bias,
                rank=max_rank,
                scale=scale,
                dtype=child.weight.dtype,
                device=child.weight.device,
            )

            new_lora.weight = child.weight
            new_lora.bias = child.bias if child.bias is not None else None

            setattr(module, name, new_lora)
        else:
            replace_linear_with_lora(
                module=child,
                max_rank=max_rank,
                scale=scale,
            )


class LinearLora(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
        rank: int,
        dtype: torch.dtype,
        device: torch.device,
        lora_bias: bool = True,
        scale: float = 1.0,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias is not None,
            device=device,
            dtype=dtype,
            *args,
            **kwargs,
        )

        assert isinstance(scale, float), "scale must be a float"

        self.scale = scale
        self.rank = rank
        self.lora_bias = lora_bias
        self.dtype = dtype
        self.device = device

        if rank > (new_rank := min(self.out_features, self.in_features)):
            self.rank = new_rank

        self.lora_A = nn.Linear(
            in_features=in_features,
            out_features=self.rank,
            bias=False,
            dtype=dtype,
            device=device,
        )
        self.lora_B = nn.Linear(
            in_features=self.rank,
            out_features=out_features,
            bias=self.lora_bias,
            dtype=dtype,
            device=device,
        )
        
        nn.init.zeros_(self.lora_B.weight)
        if self.lora_B.bias is not None:
            nn.init.zeros_(self.lora_B.bias)

    def set_scale(self, scale: float) -> None:
        assert isinstance(scale, float), "scalar value must be a float"
        self.scale = scale

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        base_out = super().forward(input)

        _lora_out_B = self.lora_B(self.lora_A(input))
        lora_update = _lora_out_B * self.scale

        return base_out + lora_update
        

class MixtureOfLoRAExperts(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_experts: int,
        rank: int,
        bias: bool = True,
        dtype: torch.dtype = None,
        device: torch.device = None,
        scale: float = 1.0,
        top_k: int = 2,  # 选择前k个专家
    ) -> None:
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            device=device,
            dtype=dtype,
        )
        
        self.num_experts = num_experts
        self.rank = min(rank, min(in_features, out_features))
        self.scale = scale
        self.top_k = min(top_k, num_experts)
        
        # 共享LoRA模块
        self.shared_lora_A = nn.Linear(in_features, self.rank, bias=False, dtype=dtype, device=device)
        self.shared_lora_B = nn.Linear(self.rank, out_features, bias=False, dtype=dtype, device=device)
        
        # 专家LoRA模块
        self.expert_lora_A = nn.ModuleList([
            nn.Linear(in_features, self.rank, bias=False, dtype=dtype, device=device)
            for _ in range(num_experts)
        ])
        self.expert_lora_B = nn.ModuleList([
            nn.Linear(self.rank, out_features, bias=False, dtype=dtype, device=device)
            for _ in range(num_experts)
        ])
        
        # 门控网络
        self.gate = nn.Linear(in_features, num_experts, dtype=dtype, device=device)
        
        # 初始化
        self._init_weights()
        
    def _init_weights(self) -> None:
        # 初始化共享LoRA
        nn.init.zeros_(self.shared_lora_B.weight)
        
        # 初始化专家LoRA
        for expert_B in self.expert_lora_B:
            nn.init.zeros_(expert_B.weight)
            
        # 初始化门控网络
        nn.init.zeros_(self.gate.bias)
        nn.init.normal_(self.gate.weight, std=0.01)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        batch_size = input.shape[0]
        
        # 基础输出 (与原始Linear层相同)
        base_out = super().forward(input)
        
        # 共享LoRA输出
        shared_lora = self.shared_lora_B(self.shared_lora_A(input))
        
        # 计算门控权重
        gate_logits = self.gate(input)
        gate_weights = F.softmax(gate_logits, dim=-1)
        
        # 选择top-k专家
        top_k_weights, top_k_indices = torch.topk(gate_weights, self.top_k, dim=-1)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        
        # 计算专家输出
        expert_outputs = torch.zeros_like(base_out)
        for k in range(self.top_k):
            # 获取当前批次中每个样本选中的专家索引
            expert_idx = top_k_indices[:, k]
            expert_weight = top_k_weights[:, k].unsqueeze(-1)
            
            # 为每个样本单独计算选中专家的输出
            for i in range(batch_size):
                idx = expert_idx[i]
                expert_out = self.expert_lora_B[idx](self.expert_lora_A[idx](input[i:i+1]))
                expert_outputs[i:i+1] += expert_out * expert_weight[i]
        
        # 组合所有输出
        final_output = (
            base_out + 
            self.scale * (shared_lora + expert_outputs)
        )
        
        return final_output

    def set_scale(self, scale: float) -> None:
        assert isinstance(scale, float), "scale must be a float"
        self.scale = scale