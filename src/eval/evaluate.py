import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import torch
import json
import argparse
from transformer_lens import HookedTransformer
from src.sae.model import TopKSAE
from src.intervention.hook import get_ablation_hook
from tqdm import tqdm

def calculate_score(model, prompts, references, device):
    matches = 0
    total = len(prompts)
    results = []
    
    print(f"Evaluating {total} prompts...")
    
    for i, prompt in enumerate(tqdm(prompts)):
        input_ids = model.to_tokens(prompt)
        
        # Generate with more descriptive parameters (Longer + Sampling to surface more knowledge)
        with torch.no_grad():
            output = model.generate(
                input_ids, 
                max_new_tokens=100, 
                do_sample=True, 
                temperature=0.8,
                top_p=0.9,
                verbose=False
            )
            
        # Extract completion
        generated_ids = output[0, input_ids.shape[1]:]
        completion = model.to_string(generated_ids)
        
        HP_KEYWORDS = [
            "harry", "potter", "ron", "weasley", "hermione", "granger", "dumbledore", "voldemort", 
            "hogwarts", "gryffindor", "slytherin", "hufflepuff", "ravenclaw", "snape", "malfoy", 
            "hagrid", "sirius", "lupin", "neville", "mcgonagall", "quidditch", "snitch", "bludger", 
            "quaffle", "muggle", "squib", "dementor", "patronus", "horcrux", "azkaban", "diagon alley", 
            "hogsmeade", "butterbeer", "gringotts", "polyjuice", "felix felicis", "basilisk", "dobby", 
            "hedwig", "fawkes", "death eater", "auror", "daily prophet", "ollivander", "platform 9", 
            "muggle-born", "mudblood", "parselmouth", "lumos", "alohomora", "expelliarmus", "cruci", 
            "imperi", "avada kedavra", "expecto patronum", "sectumsempra", "riddikulus", "wingardium", 
            "dursley", "privet drive", "bellatrix", "lestrange", "lucius", "draco", "nimbus", 
            "firebolt", "triwizard", "pensieve", "marauder", "room of requirement", "chamber of secrets",
            "wizard", "witch", "wand", "spell", "magic", "potion", "cauldron", "broomstick", "headmaster",
            "ministry of magic", "dark arts", "herbology", "potions class", "transfiguration", "charms",
            "owl post", "sorting hat", "great hall", "forbidden forest", "burrow", "godric's hollow",
            "shrieking shack", "whomping willow", "floo powder", "portkey", "marauder's map", "pensieve",
            "remembrall", "sneakoscope", "invisibility cloak", "elder wand", "resurrection stone",
            "sorting", "house points", "prefect", "quidditch pitch", "seeker", "beater", "keeper", "chaser"
        ]
        
        # Check if the completion contains NEW Harry Potter knowledge
        prompt_lower = prompt.lower()
        comp_lower = completion.lower()
        
        found = False
        for kw in HP_KEYWORDS:
            # If the keyword is in the completion but NOT in the prompt
            if kw in comp_lower and kw not in prompt_lower:
                found = True
                break
        
        if found:
            matches += 1
        
        results.append({
            "prompt": prompt,
            "completion": completion,
            "matched": found
        })
            
    return matches / total, results

def evaluate(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Load Model
    print("Loading Model...")
    model = HookedTransformer.from_pretrained("gpt2-small", device=device)
    
    # 2. Load Prompts
    prompts_path = "8940_Who_s_Harry_Potter_Approx_Supplementary Material/Eval completion prompts.json"
    if not os.path.exists(prompts_path):
        print(f"Error: Prompts file {prompts_path} not found.")
        return
         
    with open(prompts_path, "r") as f:
        data = json.load(f)
        
    prompts = [item["prompt"]["prompt"] for item in data]
    references = [item["prompt"]["references"] for item in data]
    
    # Subset for speed if needed
    if args.limit:
        prompts = prompts[:args.limit]
        references = references[:args.limit]
    
    # Baseline Eval
    baseline_score = 0
    baseline_results = []
    if not args.skip_baseline:
        print("Evaluating Baseline Model...")
        baseline_score, baseline_results = calculate_score(model, prompts, references, device)
        print(f"Baseline Match Rate: {baseline_score:.4f}")
    
    # 3. Load SAE and Features
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
    
    feature_indices = []
    mean_activations = None
    
    # Load mean activations
    mean_act_path = f"results/mean_acts_layer_{args.layer}.pt"
    if os.path.exists(mean_act_path):
        mean_activations = torch.load(mean_act_path, map_location=device)
        print("Mean activations loaded.")
    
    # Load features to ablate
    diff_means_path = f"results/layer_{args.layer}_features.pt"
    if os.path.exists(diff_means_path):
        diff_data = torch.load(diff_means_path, map_location=device)
        feature_indices = diff_data["indices"][:100] # Use top 100 features
    else:
        print(f"Warning: Features file {diff_means_path} not found.")
        
    if len(feature_indices) > 0:
        print(f"Ablating {len(feature_indices)} features...")
        print(f"Ablation scale: {args.ablation_scale}")
        
        # 4. Apply Hook with negative scaling
        hook_fn = get_ablation_hook(
            sae, 
            feature_indices, 
            mean_activations=mean_activations,
            scale=args.ablation_scale
        )
        
        # 5. Evaluate with Ablation
        print("Evaluating Ablated Model...")
        with model.hooks(fwd_hooks=[(f"blocks.{args.layer}.hook_resid_post", hook_fn)]):
            ablated_score, ablated_results = calculate_score(model, prompts, references, device)
            
        print(f"\n{'='*50}")
        print(f"Ablated Match Rate: {ablated_score:.4f}")
        
        if not args.skip_baseline:
            reduction = baseline_score - ablated_score
            rel_reduction = (reduction / baseline_score * 100) if baseline_score > 0 else 0
            print(f"Baseline Match Rate: {baseline_score:.4f}")
            print(f"Absolute Reduction: {reduction:.4f}")
            print(f"Relative Reduction: {rel_reduction:.1f}%")
        print(f"{'='*50}")
        
        # Save detailed results
        results_path = f"results/layer_{args.layer}_eval_results.json"
        output = {
            "ablated_score": ablated_score,
            "num_features_ablated": len(feature_indices),
            "ablation_scale": args.ablation_scale,
            "ablated_results": ablated_results,
        }
        if not args.skip_baseline:
            output["baseline_score"] = baseline_score
            output["baseline_results"] = baseline_results
            
        with open(results_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Saved detailed results to {results_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=8)
    parser.add_argument("--expansion_factor", type=int, default=16)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--limit", type=int, default=50, help="Limit number of prompts for speed")
    parser.add_argument("--skip_baseline", action="store_true")
    parser.add_argument("--ablation_scale", type=float, default=-5.0, 
                        help="Negative scaling factor for ablation (default -5.0 = conditional negative scaling)")
    args = parser.parse_args()
    evaluate(args)
