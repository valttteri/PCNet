import random
from datasets import load_dataset
from tqdm import tqdm

class UnifiedDataLoader:
    
    @staticmethod
    def load_test_data(dataset_name, subset=None, split="data", max_samples=500, seed=42, return_refs=False):
        """Returns a randomized list of tuples: (text, label) or (text, label, ref_answer) if return_refs=True."""
        factuals, hallucinations, refs_f, refs_h = UnifiedDataLoader._load_and_parse(
            dataset_name, subset, max_samples, seed, is_train=False
        )

        # Combine and label: 0 for Factual, 1 for Hallucinated
        if return_refs:
            combined = [(text, 0, ref) for text, ref in zip(factuals, refs_f)] + \
                       [(text, 1, ref) for text, ref in zip(hallucinations, refs_h)]
        else:
            combined = [(text, 0) for text in factuals] + [(text, 1) for text in hallucinations]
        random.shuffle(combined)
        return combined

    @staticmethod
    def load_train_data(dataset_name, subset=None, split="data", max_samples=500, seed=42):
        """Returns a tuple of two lists: (factual_texts, hallucinated_texts)"""
        factuals, hallucinations, _, _ = UnifiedDataLoader._load_and_parse(
            dataset_name, subset, max_samples, seed, is_train=True
        )
        return factuals, hallucinations

    @staticmethod
    def _load_and_parse(dataset_name, subset, max_samples, seed, is_train):
        # 🔒 FORCE DETERMINISM: Reset the seed every time this function is called.
        # This guarantees the random negative swapping is 100% identical 
        # for all baselines, models, and scripts.
        random.seed(seed)
        
        dataset_name_lower = dataset_name.lower()
        
        if dataset_name_lower in ["squad_v2", "squad", "coqa", "natural-questions", "nq_open", "copenlu/tydiqa-en"]:
            subset = None 
        
        # 1. Dynamic Split Mapping (Aligned with Haloscope specs)
        if "halueval" in dataset_name_lower:
            base_split = "data"
        elif "truthful_qa" in dataset_name_lower:
            base_split = "validation" 
        elif "coqa" in dataset_name_lower:
            base_split = "validation" 
        elif "trivia_qa" in dataset_name_lower:
            base_split = "validation" 
        elif "tydiqa" in dataset_name_lower:
            base_split = "train" # Most TyDiQA English splits use train/validation
        else:
            base_split = "validation" 

        # 2. Haloscope Split Logic:
        if is_train:
            # Grab everything EXCEPT the 25% test set
            split_slice = f"{base_split}[:-25%]"
        else:
            # Test gets exactly the reserved 25%
            split_slice = f"{base_split}[-25%:]"
            
        print(f"📥 Downloading dataset: '{dataset_name}' (subset: '{subset}', split: '{split_slice}')...")
        try:
            dataset = load_dataset(dataset_name, subset, split=split_slice)
            
            # Manually drop the first 100 items (validation buffer) for training splits
            if is_train and len(dataset) > 100:
                dataset = dataset.select(range(100, len(dataset)))
                
        except Exception as e:
            print(f"⚠️ Failed to load {dataset_name} with slice {split_slice}. Error: {e}")
            return [], [], [], []

        # Lock the HuggingFace shuffle to the same seed
        dataset = dataset.shuffle(seed=seed)
        
        factuals = []
        hallucinations = []
        refs_factual = []       # ground-truth reference answer per factual sample
        refs_hallucinated = []  # ground-truth reference answer per hallucinated sample

        # 3. Handle Sample Limits (Haloscope strict 25% rule)
        if max_samples is None:
            target_half = float('inf')
        else:
            target_half = max_samples // 2

        print(f"⚙️ Formatting {dataset_name} into Factual/Hallucinated pairs...")
        
        # ==========================================
        # DATASET PARSING ROUTINES
        # ==========================================
        
        # 1. TRUTHFUL QA (Generation Track)
        if "truthful_qa" in dataset_name_lower:
            for row in dataset:
                if len(factuals) >= target_half and len(hallucinations) >= target_half: break
                q = row.get("question", "")
                
                # Check for multiple choice
                targets = row.get("mc1_targets")
                if not targets:
                    targets = row.get("mc2_targets")
                
                if targets and "choices" in targets and "labels" in targets:
                    choices = targets["choices"]
                    labels = targets["labels"]
                    
                    correct_idx, incorrect_idx = -1, -1
                    for j, lbl in enumerate(labels):
                        val = str(lbl).strip().lower()
                        if val in ["1", "1.0", "true"] and correct_idx == -1:
                            correct_idx = j
                        elif val in ["0", "0.0", "false"] and incorrect_idx == -1:
                            incorrect_idx = j
                            
                    if len(factuals) < target_half and correct_idx != -1:
                        factuals.append(f"Question: {q}\nAnswer: {choices[correct_idx]}")
                        refs_factual.append(choices[correct_idx])
                    if len(hallucinations) < target_half and incorrect_idx != -1:
                        hallucinations.append(f"Question: {q}\nAnswer: {choices[incorrect_idx]}")
                        refs_hallucinated.append(choices[correct_idx] if correct_idx != -1 else "")

                # Handle Haloscope's Generation Track
                elif "correct_answers" in row and "incorrect_answers" in row:
                    ans_c = row["correct_answers"]
                    ans_i = row["incorrect_answers"]

                    if ans_c and len(ans_c) > 0 and len(factuals) < target_half:
                        factuals.append(f"Question: {q}\nAnswer: {ans_c[0]}")
                        refs_factual.append(ans_c[0])
                    if ans_i and len(ans_i) > 0 and len(hallucinations) < target_half:
                        hallucinations.append(f"Question: {q}\nAnswer: {ans_i[0]}")
                        refs_hallucinated.append(ans_c[0] if ans_c else "")

        # 2. SQUAD V2 & TYDIQA (Extractive QA)
        elif "squad" in dataset_name_lower or "tydiqa" in dataset_name_lower:
            valid_rows = [r for r in dataset if r.get("answers") and len(r["answers"].get("text", [])) > 0]
            
            for i, row in enumerate(valid_rows):
                if len(factuals) >= target_half and len(hallucinations) >= target_half: break
                
                # Handle varying dataset keys (context vs no context)
                ctx = row.get('context', '')
                base_str = f"Context: {ctx}\nQuestion: {row['question']}" if ctx else f"Question: {row['question']}"
                true_ans = row["answers"]["text"][0]
                
                if len(factuals) < target_half:
                    factuals.append(f"{base_str}\nAnswer: {true_ans}")
                    refs_factual.append(true_ans)

                if len(hallucinations) < target_half:
                    # Deterministic random selection
                    random_idx = random.choice([x for x in range(len(valid_rows)) if x != i])
                    fake_ans = valid_rows[random_idx]["answers"]["text"][0]
                    hallucinations.append(f"{base_str}\nAnswer: {fake_ans}")
                    refs_hallucinated.append(true_ans)

        # 3. TRIVIA QA
        elif "trivia_qa" in dataset_name_lower:
            valid_rows = [r for r in dataset if r.get("answer") and r["answer"].get("value")]
            for i, row in enumerate(valid_rows):
                if len(factuals) >= target_half and len(hallucinations) >= target_half: break
                
                q = row["question"]
                true_ans = row["answer"]["value"]
                
                if len(factuals) < target_half:
                    factuals.append(f"Question: {q}\nAnswer: {true_ans}")
                    refs_factual.append(true_ans)

                if len(hallucinations) < target_half:
                    random_idx = random.choice([x for x in range(len(valid_rows)) if x != i])
                    fake_ans = valid_rows[random_idx]["answer"]["value"]
                    hallucinations.append(f"Question: {q}\nAnswer: {fake_ans}")
                    refs_hallucinated.append(true_ans)

        # 4. COQA
        elif "coqa" in dataset_name_lower:
            valid_rows = [r for r in dataset if r.get("questions") and r.get("answers") and r["answers"].get("input_text")]
            for i, row in enumerate(valid_rows):
                if len(factuals) >= target_half and len(hallucinations) >= target_half: break
                
                base_str = f"Context: {row['story']}\nQuestion: {row['questions'][0]}"
                true_ans = row["answers"]["input_text"][0]
                
                if len(factuals) < target_half:
                    factuals.append(f"{base_str}\nAnswer: {true_ans}")
                    refs_factual.append(true_ans)

                if len(hallucinations) < target_half:
                    random_idx = random.choice([x for x in range(len(valid_rows)) if x != i])
                    fake_ans = valid_rows[random_idx]["answers"]["input_text"][0]
                    hallucinations.append(f"{base_str}\nAnswer: {fake_ans}")
                    refs_hallucinated.append(true_ans)

        # 5. HALUEVAL & NQ_OPEN 
        # (Kept in code for backward compatibility if you test them later)
        elif "halueval" in dataset_name_lower:
            for row in dataset:
                if len(factuals) >= target_half and len(hallucinations) >= target_half: break
                ctx = row.get("knowledge", "")
                q = row.get("question", "")
                base_str = f"Context: {ctx}\nQuestion: {q}" if ctx else f"Question: {q}"
                if len(factuals) < target_half:
                    factuals.append(f"{base_str}\nAnswer: {row['right_answer']}")
                    refs_factual.append(row['right_answer'])
                if len(hallucinations) < target_half:
                    hallucinations.append(f"{base_str}\nAnswer: {row['hallucinated_answer']}")
                    refs_hallucinated.append(row['right_answer'])
                    
        elif "natural-questions" in dataset_name_lower or "nq_open" in dataset_name_lower:
            for i, row in enumerate(dataset):
                if len(factuals) >= target_half and len(hallucinations) >= target_half: break
                q = row.get("query", row.get("question", ""))
                ans_data = row.get("answer", row.get("answers", []))
                
                if isinstance(ans_data, list) and len(ans_data) > 0: true_ans = ans_data[0]
                elif isinstance(ans_data, str) and ans_data.strip(): true_ans = ans_data
                else: continue 
                    
                if len(factuals) < target_half:
                    factuals.append(f"Question: {q}\nAnswer: {true_ans}")
                    refs_factual.append(true_ans)
                if len(hallucinations) < target_half:
                    random_idx = random.choice([x for x in range(len(dataset)) if x != i])
                    fake_ans_data = dataset[random_idx].get("answer", dataset[random_idx].get("answers", []))
                    fake_ans = fake_ans_data[0] if (isinstance(fake_ans_data, list) and len(fake_ans_data) > 0) else "I do not know."
                    hallucinations.append(f"Question: {q}\nAnswer: {fake_ans}")
                    refs_hallucinated.append(true_ans)

        print(f"✅ Successfully loaded 1:1 balanced pairs: {len(factuals)} factuals and {len(hallucinations)} hallucinations.")
        return factuals, hallucinations, refs_factual, refs_hallucinated