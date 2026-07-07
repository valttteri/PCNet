import random
from tqdm import tqdm
from datasets import load_dataset

class BaseParser:
    """Abstract Base Parser for extracting Factual (0) and Hallucinated (1) pairs."""
    def parse_factual(self, row, all_rows):
        raise NotImplementedError
        
    def parse_hallucinated(self, row, all_rows):
        raise NotImplementedError

    def format_output(self, context, question, answer):
        """Standardizes the output string format."""
        if context:
            return f"Context: {context}\nQuestion: {question}\nAnswer: {answer}"
        return f"Question: {question}\nAnswer: {answer}"


class HaluEvalParser(BaseParser):
    def __init__(self, subset=None):
        self.subset = subset
        
    def parse_factual(self, row, all_rows):
        if self.subset == "summarization":
            context = row.get("document", "")
            query = "Summarize the following document."
            ans = row.get("right_summary", "")
        else:
            context = row.get("knowledge", "")
            query = row.get("question", "")
            ans = row.get("right_answer", "")
        if not ans: return None
        return self.format_output(context, query, ans)
        
    def parse_hallucinated(self, row, all_rows):
        if self.subset == "summarization":
            context = row.get("document", "")
            query = "Summarize the following document."
            ans = row.get("hallucinated_summary", "")
        else:
            context = row.get("knowledge", "")
            query = row.get("question", "")
            ans = row.get("hallucinated_answer", "")
        if not ans: return None
        return self.format_output(context, query, ans)


class TruthfulQAParser(BaseParser):
    def parse_factual(self, row, all_rows):
        q = row.get("question", "")
        ans_list = row.get("correct_answers", [])
        if not ans_list: return None
        return self.format_output(None, q, ans_list[0])

    def parse_hallucinated(self, row, all_rows):
        q = row.get("question", "")
        ans_list = row.get("incorrect_answers", [])
        if not ans_list: return None
        return self.format_output(None, q, ans_list[0])


class SQuADv2Parser(BaseParser):
    def parse_factual(self, row, all_rows):
        ans_dict = row.get("answers", {})
        ans_list = ans_dict.get("text", []) if ans_dict else []
        if not ans_list: return None
        return self.format_output(row.get("context", ""), row.get("question", ""), ans_list[0])

    def parse_hallucinated(self, row, all_rows):
        # SQuAD v2 uses 'plausible_answers' for unanswerable questions
        ans_dict = row.get("plausible_answers", {})
        ans_list = ans_dict.get("text", []) if ans_dict else []
        if not ans_list: return None
        return self.format_output(row.get("context", ""), row.get("question", ""), ans_list[0])


class CoQAParser(BaseParser):
    def parse_factual(self, row, all_rows):
        story = row.get("story", "")
        questions = row.get("questions", [])
        answers = row.get("answers", {}).get("input_text", [])
        if not questions or not answers: return None
        return self.format_output(story, questions[0], answers[0])

    def parse_hallucinated(self, row, all_rows):
        story = row.get("story", "")
        questions = row.get("questions", [])
        if not questions: return None
        q = questions[0]
        
        # Negative mapping
        rand_row = random.choice(all_rows)
        fake_ans_list = rand_row.get("answers", {}).get("input_text", [])
        if not fake_ans_list: return None
        return self.format_output(story, q, fake_ans_list[0])


class TriviaQAParser(BaseParser):
    def parse_factual(self, row, all_rows):
        q = row.get("question", "")
        ans_dict = row.get("answer", {})
        val = ans_dict.get("value", "") if ans_dict else ""
        if not val: return None
        return self.format_output(None, q, val)

    def parse_hallucinated(self, row, all_rows):
        q = row.get("question", "")
        rand_row = random.choice(all_rows)
        ans_dict = rand_row.get("answer", {})
        val = ans_dict.get("value", "") if ans_dict else ""
        if not val: return None
        return self.format_output(None, q, val)


class NQOpenParser(BaseParser):
    def parse_factual(self, row, all_rows):
        q = row.get("question", "")
        ans_list = row.get("answer", [])
        if not ans_list: return None
        return self.format_output(None, q, ans_list[0])

    def parse_hallucinated(self, row, all_rows):
        q = row.get("question", "")
        rand_row = random.choice(all_rows)
        fake_ans_list = rand_row.get("answer", [])
        if not fake_ans_list: return None
        return self.format_output(None, q, fake_ans_list[0])


class UnifiedDataLoader:
    PARSERS = {
        "pminervini/HaluEval": HaluEvalParser,
        "truthful_qa": TruthfulQAParser,
        "squad_v2": SQuADv2Parser,
        "coqa": CoQAParser,
        "trivia_qa": TriviaQAParser,
        "nq_open": NQOpenParser
    }

    @classmethod
    def load_and_split(cls, dataset_name, subset=None, split="train", seed=42):
        print(f"📥 Downloading dataset: '{dataset_name}' (subset: '{subset}', split: '{split}')...")
        try:
            if subset:
                raw_ds = load_dataset(dataset_name, subset, split=split)
            else:
                raw_ds = load_dataset(dataset_name, split=split)
        except ValueError:
            fallback = "validation" if split in ["train", "data"] else "train"
            print(f"⚠️ Split '{split}' failed (likely nonexistent). Trying fallback '{fallback}'...")
            if subset:
                raw_ds = load_dataset(dataset_name, subset, split=fallback)
            else:
                raw_ds = load_dataset(dataset_name, split=fallback)
                
        return raw_ds.train_test_split(test_size=0.2, seed=seed)

    @classmethod
    def _parse_dataset(cls, dataset_name, raw_split, max_samples, subset=None):
        parser_cls = cls.PARSERS.get(dataset_name, HaluEvalParser)
        parser = parser_cls(subset) if dataset_name == "pminervini/HaluEval" else parser_cls()
        
        # We need all_rows for negative sampling if it's used
        all_rows = list(raw_split.select(range(min(max_samples * 15, len(raw_split)))))
        
        factuals = []
        hallucinated = []
        
        print(f"⚙️ Formatting {dataset_name} data into Factual/Hallucinated pairs...")
        for row in tqdm(all_rows, desc="Parsing pairs"):
            if len(factuals) >= max_samples and len(hallucinated) >= max_samples:
                break
                
            f_text = parser.parse_factual(row, all_rows)
            h_text = parser.parse_hallucinated(row, all_rows)
            
            if f_text and len(factuals) < max_samples:
                factuals.append((f_text, 0))
            if h_text and len(hallucinated) < max_samples:
                hallucinated.append((h_text, 1))
                
        # Force exact 1:1 balance mapping to the minimum of either valid lists
        min_len = min(len(factuals), len(hallucinated))
        final_factuals = factuals[:min_len]
        final_halluc = hallucinated[:min_len]
        
        print(f"✅ Successfully loaded 1:1 balanced pairs: {min_len} factuals and {min_len} hallucinations.")
        return final_factuals, final_halluc

    @classmethod
    def load_test_data(cls, dataset_name, subset=None, split="data", max_samples=500, seed=42):
        splits = cls.load_and_split(dataset_name, subset, split, seed=seed)
        test_split = splits['test']
        f_pairs, h_pairs = cls._parse_dataset(dataset_name, test_split, max_samples, subset)
        
        dataset = f_pairs + h_pairs
        random.seed(seed)
        random.shuffle(dataset)
        return dataset

    @classmethod
    def load_train_data(cls, dataset_name, subset=None, split="data", max_samples=2000, seed=42):
        splits = cls.load_and_split(dataset_name, subset, split, seed=seed)
        train_split = splits['train']
        f_pairs, h_pairs = cls._parse_dataset(dataset_name, train_split, max_samples, subset)
        
        true_texts = [p[0] for p in f_pairs]
        fake_texts = [p[0] for p in h_pairs]
        return true_texts, fake_texts
