"""
Simple Evaluation Script for Multiple Runs
==========================================

This script evaluates multiple output files (from different seeds)
and computes mean ± std and 95% confidence intervals.

Usage:
    python eval_multiple_runs.py --files output_seed0.txt output_seed1.txt output_seed2.txt
    
    OR
    
    python eval_multiple_runs.py --pattern "recommendation_output_seed*.txt"
    
    OR (if files are named with seed numbers)
    
    python eval_multiple_runs.py --prefix recommendation_output_seed --seeds 0,1,2,3,4
"""

import os
import re
import html
import glob
import argparse
import numpy as np
from scipy import stats
from difflib import SequenceMatcher


# =============================================================================
# PARSING FUNCTIONS (from your eval.py)
# =============================================================================

def normalize_title(s: str) -> str:
    if s is None:
        return ""
    s = html.unescape(s)
    s = s.strip()
    while len(s) >= 2 and ((s[0] == s[-1] and s[0] in "\"'") or (s[0], s[-1]) in [(""","""), ("'","'")]):
        s = s[1:-1].strip()
    s = s.strip("\"'""''")
    s = s.lower()
    s = s.replace("–", "-").replace("—", "-")
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\s*-\s*", "-", s)
    s = re.sub(r"\s*/\s*", "/", s)
    return s.strip()


def extract_value(line: str, prefix: str) -> str:
    val = line[len(prefix):].strip()
    val = val.replace('"item title" :', '').replace('"Item Title" :', '').strip()
    if '"' in val:
        q = [i for i, c in enumerate(val) if c == '"']
        if len(q) >= 2:
            val = val[q[0] + 1 : q[-1]]
        else:
            val = val.replace('"', '')
    return val.strip()


def get_answers_predictions(file_path):
    answers = []
    preds = []
    
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line.startswith("Answer:"):
                answers.append(normalize_title(extract_value(line, "Answer:")))
            elif line.startswith("Generated:"):
                preds.append(normalize_title(extract_value(line, "Generated:")))
            elif line.startswith("LLM:"):
                preds.append(normalize_title(extract_value(line, "LLM:")))
    
    return answers, preds


def fuzzy_match(a: str, b: str, threshold: float = 0.90) -> bool:
    return SequenceMatcher(None, a, b).ratio() >= threshold


def compute_hit1(answers, preds, fuzzy_threshold=None):
    """Compute exact and fuzzy Hit@1"""
    n = len(answers)
    exact_hits = sum(1 for a, p in zip(answers, preds) if a == p)
    exact_hit1 = exact_hits / n if n > 0 else 0
    
    fuzzy_hit1 = None
    if fuzzy_threshold:
        fuzzy_hits = sum(1 for a, p in zip(answers, preds) if fuzzy_match(a, p, fuzzy_threshold))
        fuzzy_hit1 = fuzzy_hits / n if n > 0 else 0
    
    return exact_hit1, fuzzy_hit1


# =============================================================================
# STATISTICS
# =============================================================================

def compute_statistics(scores):
    """Compute mean, std, and 95% CI"""
    scores = np.array(scores)
    n = len(scores)
    mean = np.mean(scores)
    std = np.std(scores, ddof=1) if n > 1 else 0
    
    if n > 1:
        ci = stats.t.interval(0.95, df=n-1, loc=mean, scale=stats.sem(scores))
    else:
        ci = (mean, mean)
    
    return {
        'mean': mean,
        'std': std,
        'ci_lower': ci[0],
        'ci_upper': ci[1],
        'n': n
    }


# =============================================================================
# MAIN EVALUATION
# =============================================================================

def evaluate_multiple_files(file_list, fuzzy_threshold=0.90):
    """
    Evaluate multiple output files and compute statistics.
    """
    print(f"\n{'='*60}")
    print(f"EVALUATING {len(file_list)} OUTPUT FILES")
    print(f"{'='*60}")
    
    exact_scores = []
    fuzzy_scores = []
    
    for i, filepath in enumerate(file_list):
        if not os.path.exists(filepath):
            print(f"  WARNING: File not found: {filepath}")
            continue
            
        answers, preds = get_answers_predictions(filepath)
        
        if len(answers) == 0:
            print(f"  WARNING: No data in {filepath}")
            continue
        
        exact_hit1, fuzzy_hit1 = compute_hit1(answers, preds, fuzzy_threshold)
        exact_scores.append(exact_hit1)
        if fuzzy_hit1 is not None:
            fuzzy_scores.append(fuzzy_hit1)
        
        print(f"  Run {i+1}: {os.path.basename(filepath)}")
        print(f"         Exact Hit@1: {exact_hit1:.4f}, Fuzzy Hit@1: {fuzzy_hit1:.4f}")
    
    if len(exact_scores) == 0:
        print("\nERROR: No valid files found!")
        return
    
    # Compute statistics
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    
    exact_stats = compute_statistics(exact_scores)
    print(f"\nExact Hit@1:")
    print(f"  Mean ± Std: {exact_stats['mean']:.4f} ± {exact_stats['std']:.4f}")
    print(f"  95% CI: [{exact_stats['ci_lower']:.4f}, {exact_stats['ci_upper']:.4f}]")
    print(f"  Number of runs: {exact_stats['n']}")
    
    if len(fuzzy_scores) > 0:
        fuzzy_stats = compute_statistics(fuzzy_scores)
        print(f"\nFuzzy Hit@1 (threshold={fuzzy_threshold}):")
        print(f"  Mean ± Std: {fuzzy_stats['mean']:.4f} ± {fuzzy_stats['std']:.4f}")
        print(f"  95% CI: [{fuzzy_stats['ci_lower']:.4f}, {fuzzy_stats['ci_upper']:.4f}]")
    
    # Print for paper
    print(f"\n{'='*60}")
    print("FOR YOUR PAPER")
    print(f"{'='*60}")
    print(f"""
Table: A-LLMRec Performance ({exact_stats['n']} runs)
════════════════════════════════════════════════════════
Metric              Value               95% CI
────────────────────────────────────────────────────────
Exact Hit@1         {exact_stats['mean']:.4f} ± {exact_stats['std']:.4f}     [{exact_stats['ci_lower']:.4f}, {exact_stats['ci_upper']:.4f}]""")
    
    if len(fuzzy_scores) > 0:
        print(f"Fuzzy Hit@1         {fuzzy_stats['mean']:.4f} ± {fuzzy_stats['std']:.4f}     [{fuzzy_stats['ci_lower']:.4f}, {fuzzy_stats['ci_upper']:.4f}]")
    
    print("════════════════════════════════════════════════════════")
    
    return exact_scores, fuzzy_scores


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate multiple A-LLMRec output files and compute statistics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate specific files
  python eval_multiple_runs.py --files output_seed0.txt output_seed1.txt output_seed2.txt
  
  # Use glob pattern
  python eval_multiple_runs.py --pattern "recommendation_output_seed*.txt"
  
  # Use prefix + seeds
  python eval_multiple_runs.py --prefix ./recommendation_output_seed --suffix .txt --seeds 0,1,2,3,4
        """
    )
    
    parser.add_argument('--files', nargs='+', help='List of output files to evaluate')
    parser.add_argument('--pattern', type=str, help='Glob pattern to match files')
    parser.add_argument('--prefix', type=str, help='File prefix (used with --seeds)')
    parser.add_argument('--suffix', type=str, default='.txt', help='File suffix (default: .txt)')
    parser.add_argument('--seeds', type=str, help='Comma-separated seed numbers (used with --prefix)')
    parser.add_argument('--fuzzy', type=float, default=0.90, help='Fuzzy matching threshold')
    
    args = parser.parse_args()
    
    # Determine file list
    file_list = []
    
    if args.files:
        file_list = args.files
    elif args.pattern:
        file_list = sorted(glob.glob(args.pattern))
    elif args.prefix and args.seeds:
        seeds = [int(s.strip()) for s in args.seeds.split(',')]
        file_list = [f"{args.prefix}{seed}{args.suffix}" for seed in seeds]
    else:
        parser.print_help()
        print("\nERROR: Provide --files, --pattern, or --prefix with --seeds")
        return
    
    if len(file_list) == 0:
        print("ERROR: No files found!")
        return
    
    print(f"Found {len(file_list)} files to evaluate")
    evaluate_multiple_files(file_list, fuzzy_threshold=args.fuzzy)


if __name__ == "__main__":
    main()