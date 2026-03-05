import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import torch
from transformer_lens import HookedTransformer
from src.sae.model import TopKSAE
from src.data.preprocess import load_and_tokenize, get_neutral_corpus
import argparse
import einops
from tqdm import tqdm

def analyze(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Load Model
    print("Loading Model...")
    model = HookedTransformer.from_pretrained("gpt2-small", device=device)
    
    # 2. Load SAE
    print(f"Loading SAE for Layer {args.layer}...")
    d_model = model.cfg.d_model
    d_sae = d_model * args.expansion_factor
    sae = TopKSAE(d_in=d_model, d_sae=d_sae, k=args.k)
    
    checkpoint_path = f"checkpoints/sae_layer_{args.layer}.pt"
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint {checkpoint_path} not found.")
        return
        
    sae.load_state_dict(torch.load(checkpoint_path, map_location=device))
    sae.to(device)
    sae.eval()
    
    # 3. Load Data
    print("Loading Target Corpus (Harry Potter)...")
    target_tokens = load_and_tokenize(args.target_corpus)
    target_tokens = target_tokens[:args.max_tokens]
    
    print("Loading Neutral Corpus (WikiText-2)...")
    neutral_dataset = get_neutral_corpus(split="train")
    neutral_text = "\n".join([t for t in neutral_dataset["text"][:3000] if t.strip()])
    neutral_tokens = model.tokenizer.encode(neutral_text)[:args.max_tokens]
    
    # 4. Compute Feature Activation Statistics
    def get_feature_stats(tokens, batch_size=8, ctx_len=128):
        """
        Compute per-feature statistics using the FULL forward pass (not just encode).
        Returns:
          - activation_frequency: fraction of tokens where feature > 0
          - mean_activation: mean activation value (over tokens where feature > 0)
        """
        feature_fire_count = torch.zeros(sae.d_sae, device=device)
        feature_act_sum = torch.zeros(sae.d_sae, device=device)
        total_tokens_seen = 0
        
        # Create batches
        token_chunks = [tokens[i:i+ctx_len] for i in range(0, len(tokens) - ctx_len + 1, ctx_len)]
        
        clean_batches = []
        for i in range(0, len(token_chunks), batch_size):
            batch = token_chunks[i:i+batch_size]
            if len(batch) > 0:
                try:
                    tensor_batch = torch.tensor(batch).to(device)
                    if tensor_batch.ndim == 1:
                        tensor_batch = tensor_batch.unsqueeze(0)
                    if tensor_batch.shape[1] == ctx_len:
                        clean_batches.append(tensor_batch)
                except:
                    pass
        
        print(f"  Processing {len(clean_batches)} batches...")
        with torch.no_grad():
            for batch in tqdm(clean_batches, desc="  Features"):
                # Run model to get activations
                _, cache = model.run_with_cache(batch, stop_at_layer=args.layer + 1)
                acts = cache[f"blocks.{args.layer}.hook_resid_post"]
                
                # Use FULL forward pass (not just encode) to get post-TopK sparse activations
                _, z_sparse = sae(acts)
                # z_sparse shape: [batch, seq, d_sae]
                
                z_flat = einops.rearrange(z_sparse, "b s d -> (b s) d")
                
                # Activation frequency: how often does each feature fire (> 0)?
                fired = (z_flat > 0).float()
                feature_fire_count += fired.sum(dim=0)
                
                # Mean activation (sum of activations for computing mean later)
                feature_act_sum += z_flat.sum(dim=0)
                
                total_tokens_seen += z_flat.shape[0]
                
        activation_freq = feature_fire_count / total_tokens_seen
        mean_activation = feature_act_sum / (feature_fire_count + 1e-8)  # mean over firing tokens only
        
        return activation_freq, mean_activation, total_tokens_seen

    print("\nComputing Target Feature Statistics...")
    target_freq, target_mean_act, target_count = get_feature_stats(target_tokens)
    
    print("\nComputing Neutral Feature Statistics...")
    neutral_freq, neutral_mean_act, neutral_count = get_feature_stats(neutral_tokens)
    
    # 5. Difference in Means (using ACTIVATION FREQUENCY as primary metric)
    freq_diff = target_freq - neutral_freq
    
    # Filter: only consider features that actually fire on target corpus
    min_freq = args.min_freq
    valid_mask = target_freq > min_freq
    freq_diff_masked = freq_diff.clone()
    freq_diff_masked[~valid_mask] = -float('inf')
    
    # 6. Select Top Features
    top_vals, top_inds = torch.topk(freq_diff_masked, k=args.num_features)
    
    print(f"\n{'='*70}")
    print(f"Top {args.num_features} Harry Potter-specific features (by activation frequency difference):")
    print(f"{'='*70}")
    print(f"{'Feature':>10} | {'Target Freq':>12} | {'Neutral Freq':>12} | {'Freq Diff':>10} | {'Mean Act':>10}")
    print(f"{'-'*10}-+-{'-'*12}-+-{'-'*12}-+-{'-'*10}-+-{'-'*10}")
    
    for i in range(args.num_features):
        idx = top_inds[i].item()
        print(f"{idx:>10} | {target_freq[idx].item():>12.4f} | {neutral_freq[idx].item():>12.4f} | "
              f"{freq_diff[idx].item():>10.4f} | {target_mean_act[idx].item():>10.4f}")
        
    # 7. Save results (indices, frequency data, and mean activations for ablation)
    save_path = f"results/layer_{args.layer}_features.pt"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        "indices": top_inds,
        "freq_diff": top_vals,
        "target_freq": target_freq,
        "neutral_freq": neutral_freq,
        "target_mean_activation": target_mean_act,
        "neutral_mean_activation": neutral_mean_act,
    }, save_path)
    print(f"\nSaved selected features to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=8)
    parser.add_argument("--target_corpus", type=str, default="src/data/target_corpus.txt")
    parser.add_argument("--expansion_factor", type=int, default=16)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--num_features", type=int, default=20)
    parser.add_argument("--max_tokens", type=int, default=300000, help="Max tokens from each corpus")
    parser.add_argument("--min_freq", type=float, default=0.001, help="Min activation frequency on target")
    args = parser.parse_args()
    analyze(args)
