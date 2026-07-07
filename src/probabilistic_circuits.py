import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

# ======================================================
# JIT-Compiled Math Kernels
# ======================================================
@torch.jit.script
def log_gaussian_jit(x: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    # 0.5 * log(2*pi) ~= 0.9189385
    return -0.5 * ((x - mu) / sigma)**2 - torch.log(sigma) - 0.9189385

@torch.jit.script
def log_laplace_jit(x: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    # b = sigma / sqrt(2). log(2b) = log(sigma*sqrt(2)) = log(sigma) + 0.34657
    b = sigma * 0.70710678
    return -torch.abs(x - mu) / b - (torch.log(sigma) + 0.3465736)

@torch.jit.script
def log_student_jit(x: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor, nu: torch.Tensor) -> torch.Tensor:
    # Stable Student-T implementation
    z = (x - mu) / (sigma + 1e-12)
    term1 = torch.lgamma((nu + 1) * 0.5)
    term2 = torch.lgamma(nu * 0.5)
    term3 = 0.5 * torch.log(nu * 3.14159265)
    term4 = torch.log(sigma + 1e-12)
    term5 = ((nu + 1) * 0.5) * torch.log1p((z ** 2) / nu)
    return term1 - term2 - term3 - term4 - term5

# ======================================================
# Base Node
# ======================================================
class AbstractNode(nn.Module):
    def __init__(self, child_nodes=None):
        super().__init__()
        # Renamed from 'children' to 'child_nodes' to avoid conflict with nn.Module.children()
        self.child_nodes = nn.ModuleList(child_nodes) if child_nodes else nn.ModuleList()

    def forward(self, x):
        raise NotImplementedError

# ======================================================
# Input Node (No Changes needed, but included for completeness)
# ======================================================
class InputNode(AbstractNode):
    def __init__(self, feature_idx, mu_init=0.0, sigma_init=1.0):
        # InputNodes have no children
        super().__init__()
        self.feature_idx = feature_idx
        
        # Learnable Parameters
        self.mu = nn.Parameter(torch.tensor(float(mu_init)))
        self.log_sigma = nn.Parameter(torch.log(torch.tensor(float(sigma_init))))
        self.log_nu = nn.Parameter(torch.log(torch.tensor(5.0))) 
        self.logits = nn.Parameter(torch.zeros(3)) 
        self.gate = nn.Parameter(torch.tensor(2.0)) 

    def fit(self, X):
        """Statistical initialization"""
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        vals = X[:, self.feature_idx].reshape(-1)
        
        mu = float(np.median(vals))
        mad = float(np.median(np.abs(vals - mu)) + 1e-6)
        sigma = mad * 1.4826
        
        with torch.no_grad():
            self.mu.fill_(mu)
            self.log_sigma.fill_(np.log(sigma))
            self.log_nu.fill_(np.log(5.0))
            self.logits.fill_(0.0)

    def forward(self, x):
        vals = x[:, self.feature_idx]
        
        sigma = F.softplus(self.log_sigma) + 1e-6
        nu    = F.softplus(self.log_nu) + 1e-3      
        
        log_gauss = log_gaussian_jit(vals, self.mu, sigma)
        log_lapl  = log_laplace_jit(vals, self.mu, sigma)
        log_stud  = log_student_jit(vals, self.mu, sigma, nu)
        
        # Determine the target shape dynamically
        # If input is (N, C), vals is (N,). Target: (N, 1, 1)
        # If input is (B, C, H, W), vals is (B, H, W). Target: (B, 1, H, W)
        target_shape = [vals.shape[0], 1] + list(vals.shape[1:])
        
        log_gauss = log_gauss.view(*target_shape)
        log_lapl = log_lapl.view(*target_shape)
        log_stud = log_stud.view(*target_shape)

        # Vector of size 3 -> (3, 1, 1...) to match stack
        # Add 1s for Batch, Channel, and Spatial dims
        w_shape = [3] + [1] * len(target_shape)
        w = torch.log_softmax(self.logits, dim=0).view(*w_shape)
        
        # Stack is (3, B, 1, H, W) or (3, N, 1, 1)
        stack = torch.stack([log_gauss, log_lapl, log_stud], dim=0) 
            
        # Broadcasts perfectly over Mixture components (dim=0)
        log_mix = torch.logsumexp(w + stack, dim=0)
        
        return log_mix * torch.sigmoid(self.gate)

# ======================================================
# Sum Node (Updated variable name)
# ======================================================
class SumNode(AbstractNode):
    def __init__(self, child_nodes, weights=None):
        super().__init__(child_nodes)
        n = len(child_nodes)
        if weights is None:
            self.weights = nn.Parameter(torch.full((n,), -np.log(n)))
        else:
            w_t = torch.tensor(weights, dtype=torch.float32)
            self.weights = nn.Parameter(torch.log(w_t / w_t.sum()))

    def forward(self, x):
        # Iterate over self.child_nodes instead of self.children
        child_outputs = [c(x) for c in self.child_nodes]
        
        stack = torch.stack(child_outputs, dim=0)
        
        # log_w initially is (N_children,)
        # We need it to be (N_children, 1, 1...) to broadcast against stack of shape (N_children, Batch, ...)
        w_shape = [-1] + [1] * (stack.dim() - 1)
        log_w = torch.log_softmax(self.weights, dim=0).view(*w_shape)
        
        return torch.logsumexp(stack + log_w, dim=0)

# ======================================================
# Product Node (Updated variable name)
# ======================================================
class ProductNode(AbstractNode):
    def forward(self, x):
        child_outputs = [c(x) for c in self.child_nodes]
        stack = torch.stack(child_outputs, dim=0)
        return torch.sum(stack, dim=0)

# ======================================================
# Classifier / Root Node (Updated variable name)
# ======================================================
class ClassifierNode(AbstractNode):
    def forward(self, x):
        return torch.stack([c(x) for c in self.child_nodes], dim=1)

# ======================================================
# PCNet: The Graph Manager
# ======================================================
class PCNet(nn.Module):
    def __init__(self, n_classes=2, max_depth=4, max_branching=3, seed=42):
        super().__init__()
        self.n_classes = n_classes
        self.max_depth = max_depth
        self.max_branching = max_branching
        self.seed = seed
        self.root = None

    def init_network(self, inputs):
        """
        Builds the PC structure based on input data.
        """
        # Reproducibility
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        
        # 1. Create Leaves (InputNodes)
        B, C, H, W = inputs.shape if inputs.dim() == 4 else (inputs.shape[0], inputs.shape[1], 1, 1)
        
        # Create temporary list; they become registered when added to ModuleList/Node
        current_nodes = []
        for i in range(C):
            node = InputNode(i)
            node.fit(inputs) # Initialize stats (Median/MAD)
            current_nodes.append(node)
            
        # Register the leaves for the monolithic structure
        self.leaves = nn.ModuleList(current_nodes)
        
        # 2. Build Random DAG Structure (Bottom-Up)
        nodes = current_nodes
        for depth in range(1, self.max_depth + 1):
            
            # Final Layer: Reduce to N classes
            if depth == self.max_depth:
                nodes = self._reduce_to_n(nodes, self.n_classes)
                self.root = ClassifierNode(nodes)
                break
            
            # Intermediate Layers: Alternate Product & Sum
            if depth % 2 == 0:
                nodes = self._prod_layer(nodes)
            else:
                nodes = self._sum_layer(nodes)

        # Fallback if depth was too small
        if self.root is None:
             self.root = ClassifierNode(self._reduce_to_n(current_nodes, self.n_classes))

    def forward(self, x):
        if self.root is None:
            raise RuntimeError("PCNet not initialized. Call init_network(data) first.")
        return self.root(x)

    # ------------------------------------------------------------------
    # Graph Topology Helpers
    # ------------------------------------------------------------------
    def _pairwise(self, nodes):
        """Shuffle and pair nodes for product layers"""
        pool = list(nodes)
        random.shuffle(pool)
        pairs = []
        while len(pool) >= 2:
            pairs.append((pool.pop(), pool.pop()))
        if pool:
            pairs.append((pool.pop(),))
        return pairs

    def _prod_layer(self, nodes):
        """Create Product Nodes (and pass through some residuals)"""
        next_nodes = []
        for pair in self._pairwise(nodes):
            if len(pair) == 1:
                next_nodes.append(pair[0])
            else:
                next_nodes.append(ProductNode(list(pair)))
        return next_nodes

    def _sum_layer(self, nodes):
        """Create Sum Nodes with random branching factors"""
        next_nodes = []
        pool = list(nodes)
        random.shuffle(pool)
        
        while pool:
            # Random branching factor between 2 and max_branching
            k = random.randint(2, self.max_branching)
            group = []
            for _ in range(k):
                if pool: group.append(pool.pop())
            
            if len(group) == 1:
                next_nodes.append(group[0])
            else:
                s_node = SumNode(group)
                next_nodes.append(s_node)
                
                # Residual Connection: Occasionally pass a child through directly
                # This helps gradient flow in deeper networks
                if random.random() < 0.2:
                    next_nodes.append(group[0])
                    
        return next_nodes

    def _reduce_to_n(self, nodes, n):
        """Force the layer to have exactly n nodes (for the final class output)"""
        if len(nodes) < n:
            # Pad with the last node
            return nodes + [nodes[-1]] * (n - len(nodes))
        
        # Split into n groups and sum them
        chunk_size = int(np.ceil(len(nodes) / n))
        reduced = []
        for i in range(0, len(nodes), chunk_size):
            group = nodes[i : i + chunk_size]
            if len(group) == 1:
                reduced.append(group[0])
            else:
                reduced.append(SumNode(group))
        
        return reduced[:n]

# ======================================================
# Attentional PCNet Nodes
# ======================================================
class AttentionalSumNode(AbstractNode):
    def __init__(self, child_nodes, input_dim, embed_dim=16):
        super().__init__(child_nodes)
        self.n_children = len(child_nodes)
        self.embed_dim = embed_dim
        
        # KEY: Static learned embeddings for each child sub-circuit
        self.keys = nn.Parameter(torch.randn(self.n_children, embed_dim))
        
        # QUERY: Network to project the input into the query space
        self.query_proj = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        self.scale = np.sqrt(embed_dim)

    def forward(self, x):
        # 1. VALUES (V): Calculate child log-probabilities normally
        child_outputs = [c(x) for c in self.child_nodes]
        stack = torch.stack(child_outputs, dim=0) # Shape: (N_children, Batch, ...)
        
        # 2. QUERY (Q): Project the raw input
        # Assuming x is (B, C, H, W) or (B, D)
        if x.dim() == 4:
            raw_input = x.mean(dim=(2, 3)) # Global Average Pooling
        else:
            raw_input = x
            
        queries = self.query_proj(raw_input) # Shape: (Batch, embed_dim)
        
        # 3. ATTENTION SCORES: Q dot K^T
        # keys shape: (N_children, embed_dim)
        # scores shape: (Batch, N_children)
        attn_scores = torch.matmul(queries, self.keys.t()) / self.scale
        
        # Transpose to match stack: (N_children, Batch)
        attn_scores = attn_scores.t() 
        
        # Reshape for broadcasting: (N_children, Batch, 1, 1...)
        w_shape = list(attn_scores.shape) + [1] * (stack.dim() - 2)
        log_w = torch.log_softmax(attn_scores, dim=0).view(*w_shape)
        
        # 4. CONTEXT: Dynamic Mixture (LogSumExp)
        return torch.logsumexp(stack + log_w, dim=0)

class MultiHeadAttentionalSumNode(AbstractNode):
    def __init__(self, child_nodes, input_dim, embed_dim=32, num_heads=4):
        super().__init__(child_nodes)
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.n_children = len(child_nodes)
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # KEY: Learned embeddings per child, per head
        # Shape: (Num_Heads, N_children, Head_Dim)
        self.keys = nn.Parameter(torch.randn(num_heads, self.n_children, self.head_dim))
        
        # QUERY: Project input into a multi-head query space
        self.query_proj = nn.Linear(input_dim, embed_dim)
        
        # Learnable head-weights to combine the experts' opinions
        self.head_mix = nn.Parameter(torch.zeros(num_heads))
        self.scale = np.sqrt(self.head_dim)

    def forward(self, x):
        # 1. VALUES (V): Calculate child log-probabilities
        child_outputs = [c(x) for c in self.child_nodes]
        stack = torch.stack(child_outputs, dim=0) # (N_children, Batch, ...)
        
        # 2. QUERY (Q): Project and reshape into heads
        if x.dim() == 4:
            raw_input = x.mean(dim=(2, 3)) 
        else:
            raw_input = x
            
        # Shape: (Batch, Num_Heads, Head_Dim)
        queries = self.query_proj(raw_input).view(-1, self.num_heads, self.head_dim)
        
        # 3. MULTI-HEAD ATTENTION SCORES
        # queries: (Batch, Heads, Dim) -> (Heads, Batch, Dim)
        # keys:    (Heads, Children, Dim)
        # Result (attn_scores): (Heads, Batch, Children)
        queries = queries.transpose(0, 1)
        attn_scores = torch.matmul(queries, self.keys.transpose(-1, -2)) / self.scale
        
        # 4. COMPUTE LOG WEIGHTS PER HEAD
        log_w_heads = torch.log_softmax(attn_scores, dim=-1) # (Heads, Batch, Children)
        
        # 5. AGGREGATE HEADS
        # We combine head decisions via a learned mixture (LogSumExp over heads)
        # head_mix: (Heads, 1, 1) for broadcasting
        head_weights = torch.log_softmax(self.head_mix, dim=0).view(-1, 1, 1)
        
        # Final log weights for children: (Batch, Children)
        # We sum across heads to get a single distribution over children
        log_w = torch.logsumexp(log_w_heads + head_weights, dim=0)
        
        # 6. FINAL MIXTURE
        # Transpose log_w to (Children, Batch) and broadcast to stack shape
        log_w = log_w.t()
        w_shape = list(log_w.shape) + [1] * (stack.dim() - 2)
        
        return torch.logsumexp(stack + log_w.view(*w_shape), dim=0)


class AttentionalPCNet(PCNet):
    def __init__(self, in_channels, n_classes=2, max_depth=4, max_branching=3, seed=42, embed_dim=16):
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        super().__init__(n_classes=n_classes, max_depth=max_depth, max_branching=max_branching, seed=seed)

    def _sum_layer(self, nodes):
        """Create Attentional Sum Nodes with random branching factors"""
        next_nodes = []
        pool = list(nodes)
        random.shuffle(pool)
        
        while pool:
            # Random branching factor between 2 and max_branching
            k = random.randint(2, self.max_branching)
            group = []
            for _ in range(k):
                if pool: group.append(pool.pop())
            
            if len(group) == 1:
                next_nodes.append(group[0])
            else:
                s_node = AttentionalSumNode(group, input_dim=self.in_channels, embed_dim=self.embed_dim)
                next_nodes.append(s_node)
                
                # Residual Connection
                if random.random() < 0.2:
                    next_nodes.append(group[0])
                    
        return next_nodes

    def _reduce_to_n(self, nodes, n):
        """Force the layer to have exactly n nodes (for the final class output)"""
        if len(nodes) < n:
            # Pad with the last node
            return nodes + [nodes[-1]] * (n - len(nodes))
        
        # Split into n groups and sum them
        chunk_size = int(np.ceil(len(nodes) / n))
        reduced = []
        for i in range(0, len(nodes), chunk_size):
            group = nodes[i : i + chunk_size]
            if len(group) == 1:
                reduced.append(group[0])
            else:
                reduced.append(AttentionalSumNode(group, input_dim=self.in_channels, embed_dim=self.embed_dim))
        
        return reduced[:n]
    
class MultiHeadAttentionalPCNet(PCNet):
    def __init__(self, in_channels, n_classes=2, max_depth=4, 
                 max_branching=3, seed=42, embed_dim=32, num_heads=4):
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        super().__init__(n_classes=n_classes, max_depth=max_depth, 
                         max_branching=max_branching, seed=seed)

    def _sum_layer(self, nodes):
        """Create Multi-Head Attentional Sum Nodes"""
        next_nodes = []
        pool = list(nodes)
        random.shuffle(pool)
        
        while pool:
            k = random.randint(2, self.max_branching)
            group = []
            for _ in range(k):
                if pool: group.append(pool.pop())
            
            if len(group) == 1:
                next_nodes.append(group[0])
            else:
                # Injecting the Multi-Head Node here
                s_node = MultiHeadAttentionalSumNode(
                    group, 
                    input_dim=self.in_channels, 
                    embed_dim=self.embed_dim,
                    num_heads=self.num_heads
                )
                next_nodes.append(s_node)
                
                # Residual path for better gradient flow
                if random.random() < 0.2:
                    next_nodes.append(group[0])
                    
        return next_nodes

    def _reduce_to_n(self, nodes, n):
        """Final layer reduction with Multi-Head Attention"""
        if len(nodes) < n:
            return nodes + [nodes[-1]] * (n - len(nodes))
        
        chunk_size = int(np.ceil(len(nodes) / n))
        reduced = []
        for i in range(0, len(nodes), chunk_size):
            group = nodes[i : i + chunk_size]
            if len(group) == 1:
                reduced.append(group[0])
            else:
                reduced.append(MultiHeadAttentionalSumNode(
                    group, 
                    input_dim=self.in_channels, 
                    embed_dim=self.embed_dim,
                    num_heads=self.num_heads
                ))
        return reduced[:n]
    
    
