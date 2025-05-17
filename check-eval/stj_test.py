import eval_type_backport
import json
import numpy as np
from tqdm import tqdm
from utils import calculate_correlation
from checkeval.checkeval import Checkeval
from dotenv import main
import os
from argparse import ArgumentParser
import pandas as pd
from pathlib import Path

main.load_dotenv()

criteria_definitions = {
  "linguistic_fluency": {
    "natural_expression": "The writing should sound natural and maintain a smooth, fluid rhythm with a consistent tone.",
    "text_length": "The overall length should be appropriate and proportional to the depth and scope of the content.",
    "vocabulary": "Word choices should be contextually appropriate and well-suited to the subject matter.",
    "syntax": "Sentences should demonstrate proper grammar and structure, contributing to clarity and readability.",
    "mechanic_spelling_punctuation": "Spelling and punctuation should be correct and follow standard writing conventions."
  },
  "logical_fluency": {
    "organization_layout": "The structure of the writing should be coherent, with ideas presented in a logical and organized manner.",
    "repetitive_content": "Content should avoid unnecessary repetition of words, phrases, or ideas.",
    "inter_sentence_cohesion": "Sentences should connect logically and flow smoothly from one to the next.",
    "inter_paragraph_cohesion": "Paragraphs should transition clearly, maintaining logical relationships between ideas."
  },
  "coherence": {
    "topic_consistency": "The writing should maintain a consistent theme or subject throughout.",
    "topic_sentence_paragraph": "Each paragraph should be focused on a clear subtopic or aspect of the main idea."
  },
  "consistency": {
    "tone": "The tone should remain consistent throughout, whether formal, informal, neutral, or expressive.",
    "stance_posture": "The point of view or position taken should remain steady and logically developed.",
    "style": "The overall writing style should be coherent and not shift unexpectedly (e.g., from casual to technical)."
  },
  "complexity": {
    "vocabulary": "Vocabulary should reflect a thoughtful level of complexity, without being overly simplistic or needlessly complex.",
    "syntax": "Sentence structures should be varied and show an appropriate degree of sophistication."
  },
  "specificity": {
    "use_of_examples_and_review": "Relevant examples, comparisons, or evidence should be included where helpful.",
    "detailed_descriptions": "Specific facts, figures, or in-depth details should be used to enrich the content when appropriate."
  },
  "interestingness": {
     "curiosity_arousal": "The content should spark curiosity or raise intriguing questions that encourage continued reading.",
     "surprise_or_novelty": "The text should include unexpected facts, perspectives, or twists that defy reader expectations.",
     "emotional_resonance": "The writing should evoke emotional reactions such as awe, joy, sadness, or empathy."
  }
}

def flatten_criteria(criteria_dict):
    flat = {}
    for group, subs in criteria_dict.items():
        for sub, definition in subs.items():
            flat[f"{group}.{sub}"] = definition
    return flat

flat_criteria_definitions = flatten_criteria(criteria_definitions)


def compute_overall_scores(check_eval):
    from collections import defaultdict
    grouped = defaultdict(list)
    for sub, score in check_eval.items():
        main = sub.split(".")[0]
        grouped[main].append(score)
    return {main: round(np.mean(scores), 4) for main, scores in grouped.items()}


class STJ:
    def __init__(self, model="gpt-4o"):
        self.model = model
        self.client = Checkeval(
            api_key=os.environ.get("OPENAI_KEY"),
            model=self.model,
            criteria_definitions=flat_criteria_definitions
        )
        self.dataset = self.load_data()
        self.generate_checklist_prompt = open(Path(__file__).parent / "checkeval" / "prompts" / "generate_checklist.md").read()
        self.evaluate_checklist_prompt = open(Path(__file__).parent / "checkeval" / "prompts" / "evaluate_checklist.md").read()

    def load_data(self, split="TEST"):
        data = pd.read_csv("./data/human_generated_2.csv")
        data['sentence_A'] = ""
        return data

    def generate(self, criterion="linguistic_fluency", method="criterion"):
        sub_criteria = [key for key in flat_criteria_definitions if key.startswith(criterion + ".")]

        pred_scores = {sub: [] for sub in sub_criteria}
        all_results = []
        pbar = tqdm(total=len(self.dataset))
        failed_ids = []
        for i, row in self.dataset.iterrows():
            try:
                os.makedirs("results", exist_ok=True)
                row_dict = row.to_dict()
                reference = row_dict["sentence_A"]
                candidate = row_dict["sentence_B"]
                row_dict["check_eval"] = {}
                row_dict["checklists"] = {}

                for sub in sub_criteria:
                    definition = flat_criteria_definitions[sub]

                    if method == "reference":
                        res = self.client.reference_guided(sub, reference, candidate, checklist=None, criterion_definition=definition)
                    elif method == "candidate":
                        res = self.client.candidate_guided(sub, reference, candidate, checklist=None)
                    elif method == "criterion":
                        res = self.client.criterion_guided(sub, reference, candidate, checklist=None)
                    else:
                        raise ValueError("Invalid method")

                    score = res["results"].score()

                    checklist_response = res["results"].items
                    yes_items = [item for item in checklist_response if item.isChecked]
                    no_items = [item for item in checklist_response if not item.isChecked]

                    checklist_text = res["checklist"].to_markdown() if hasattr(res["checklist"], "to_markdown") else str(res["checklist"])
                    checklist_text += f"\n Yes: {len(yes_items)} |  No: {len(no_items)}\n"
                    checklist_text += "\nBreakdown:\n"

                    for item in checklist_response:
                        checklist_items = res["checklist"].items if hasattr(res["checklist"], "items") else []
                        question_text = next((q.text for q in checklist_items if q.number == item.item), f"Checklist Question {item.item}")
                        status = " Yes" if item.isChecked else " No"
                        checklist_text += f"{item.item}. {question_text} --> {status}\n"

                    pred_scores[sub].append(score)
                    row_dict["check_eval"][sub] = score
                    row_dict["checklists"][sub] = checklist_text

                row_dict["overall_eval"] = compute_overall_scores(row_dict["check_eval"])
                all_results.append(row_dict)

                with open(f"results/stj_zero_{criterion}_{method}.txt", "w") as output_file:
                    for row in all_results:
                        output_file.write("Reference: " + row.get("sentence_A", "") + "\n")
                        output_file.write("Candidate: " + row.get("sentence_B", "") + "\n")
                        for sub, checklist in row["checklists"].items():
                            output_file.write(f"Criterion: {sub}\n")
                            output_file.write(checklist + "\n")
                        output_file.write("Overall Evaluation:\n")
                        for crit, score in row["overall_eval"].items():
                            output_file.write(f"  {crit}: {score}\n")
                        output_file.write("\n" + "="*40 + "\n")

                pbar.update(1)
            except Exception as e:
                print(f"Error processing row {i}: {e}")
                failed_ids.append(i)
                pbar.update(1)
                continue

        print("\nFinal Evaluation:")
        flat_scores = [score for scores in pred_scores.values() for score in scores]
        print("Average Score:", round(np.mean(flat_scores) * 100, 2))

        summary = {
            "average_scores": {sub: round(np.mean(pred_scores[sub]) * 100, 2) for sub in sub_criteria},
            "overall_average_score": round(np.mean(flat_scores) * 100, 2)
        }

        with open(f"results/human_{criterion}_{method}_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        if failed_ids:
            with open(f"results/stj_zero_{criterion}_{method}_failed_ids.txt", "w") as f:
                f.write("Failed IDs:\n")
                for failed_id in failed_ids:
                    f.write(f"{failed_id}\n")

        print("\nSummary saved to:", f"results/human_{criterion}_{method}_summary.json")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--criterion", type=str, default="interestingness")
    parser.add_argument("--method", type=str, choices=["reference", "candidate", "criterion"], default="criterion")
    parser.add_argument("--model", type=str, default="gpt-4o")
    args = parser.parse_args()

    all_criteria = [
        "linguistic_fluency",
        "logical_fluency",
        "coherence",
        "consistency",
        "complexity",
        "specificity",
        "interestingness"
    ]

    summary_scores = {}

    for criterion in all_criteria:
        print(f"Running evaluation for criterion: {criterion}")
        stj = STJ(model=args.model)
        stj.generate(criterion, args.method)

        summary_path = f"results/human_{criterion}_{args.method}_summary.json"
        if os.path.exists(summary_path):
            with open(summary_path, "r") as f:
                summary = json.load(f)
                summary_scores[criterion] = summary.get("overall_average_score", 0)
        else:
            print(f"Summary file not found: {summary_path}")

    with open("results/human_overall_summary.txt", "w") as summary_file:
        summary_file.write("Final Summary of Overall Average Scores:\n")
        for criterion, avg_score in summary_scores.items():
            summary_file.write(f"{criterion}: {avg_score}%\n")

    print("Completed evaluations. Summary written to results/human_overall_summary.txt")
