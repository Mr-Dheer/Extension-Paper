import numpy as np
import html

def normalize_text(text):
    """Normalize text for comparison"""
    # Decode HTML entities (&amp; -> &, &quot; -> ", etc.)
    text = html.unescape(text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove all quotes
    text = text.replace('"', '').replace("'", '').replace('"', '').replace('"', '')
    
    # Strip whitespace
    text = text.strip()
    
    return text

def get_answers_predictions(file_path):
    answers = []
    llm_predictions = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            if line.startswith('Answer:'):
                answer = line.replace('Answer:', '').strip()
                answer = normalize_text(answer)
                answers.append(answer)
                
            elif line.startswith('LLM:'):
                llm_prediction = line.replace('LLM:', '').strip()
                llm_prediction = normalize_text(llm_prediction)
                llm_predictions.append(llm_prediction)
                
    return answers, llm_predictions

def evaluate(answers, llm_predictions, k=1):
    NDCG = 0.0
    HT = 0.0
    predict_num = len(answers)
    
    # Detailed debugging
    print(f"\n{'='*80}")
    print(f"EVALUATION DEBUG - First 10 comparisons:")
    print(f"{'='*80}\n")
    
    matches = []
    for i, (ans, pred) in enumerate(zip(answers[:10], llm_predictions[:10])):
        match = ans == pred
        matches.append(match)
        
        print(f"Example {i+1}: {'✓ MATCH' if match else '✗ MISS'}")
        print(f"  Answer:     '{ans}'")
        print(f"  Prediction: '{pred}'")
        
        if not match:
            # Show character-level difference
            print(f"  Lengths: {len(ans)} vs {len(pred)}")
            if len(ans) == len(pred):
                for j, (c1, c2) in enumerate(zip(ans, pred)):
                    if c1 != c2:
                        print(f"  First diff at position {j}: '{c1}' vs '{c2}'")
                        break
        print()
    
    print(f"{'='*80}")
    print(f"Sample match rate: {sum(matches)}/{len(matches)}")
    print(f"{'='*80}\n")
    
    # Full evaluation
    for answer, prediction in zip(answers, llm_predictions):
        if k > 1:
            # For top-k evaluation (if predictions were lists)
            if answer in prediction:
                rank = prediction.index(answer)
                if rank < k:
                    NDCG += 1 / np.log2(rank + 2)
                    HT += 1
        elif k == 1:
            # Exact match for top-1
            if answer == prediction:
                NDCG += 1
                HT += 1
                
    return NDCG / predict_num, HT / predict_num

if __name__ == "__main__":
    inferenced_file_path = '/home/kavach/Dev/Extension-Paper/A-LLMRec/recommendation_output.txt'
    
    print("Loading predictions...")
    answers, llm_predictions = get_answers_predictions(inferenced_file_path)
    
    print(f"Loaded {len(answers)} answers and {len(llm_predictions)} predictions")
    assert len(answers) == len(llm_predictions), \
        f"Mismatch: {len(answers)} answers vs {len(llm_predictions)} predictions"
    
    # Evaluate
    ndcg, ht = evaluate(answers, llm_predictions, k=1)
    
    # Final results
    print(f"\n{'='*80}")
    print(f"FINAL RESULTS")
    print(f"{'='*80}")
    print(f"Total samples: {len(answers)}")
    print(f"NDCG@1: {ndcg:.4f}")
    print(f"Hit@1:  {ht:.4f}")
    print(f"Accuracy: {ht*100:.2f}%")
    print(f"{'='*80}\n")
