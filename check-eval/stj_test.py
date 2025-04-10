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
        "natural_expression": "The candidate should read naturally and maintain a fluid rhythm, aligning its tone with the reference text.",
        "text_length": "The length of the candidate should be appropriate and proportional to the content in the reference text.",
        "vocabulary": "The vocabulary used in the candidate should be contextually suitable and comparable to that in the reference.",
        "syntax": "The candidate should demonstrate proper sentence structure in a manner consistent with the reference.",
        "mechanic_spelling_punctuation": "Spelling and punctuation in the candidate should be correct and on par with the reference quality."
    },
    "logical_fluency": {
        "organization_layout": "The candidate should follow a coherent structure inspired by the organization of the reference.",
        "repetitive_content": "The candidate should avoid unnecessary repetition compared to the reference.",
        "inter_sentence_cohesion": "Sentences in the candidate should connect logically as they do in the reference.",
        "inter_paragraph_cohesion": "The flow between paragraphs in the candidate should mirror the logical transitions in the reference."
    },
    "coherence": {
        "topic_consistency": "The candidate should maintain a consistent theme that aligns with the reference content.",
        "topic_sentence_paragraph": "Each paragraph in the candidate should reflect subtopics derived from the reference paragraphs."
    },
    "consistency": {
        "tone": "The tone of the candidate should be consistent with the tone observed in the reference.",
        "stance_posture": "The candidate should uphold a coherent stance that is either aligned with or reasonably derived from the reference.",
        "style": "The candidate should adopt a stylistic approach that mirrors the style of the reference (e.g., formal/informal, spoken/written)."
    },
    "complexity": {
        "vocabulary": "The vocabulary in the candidate should reflect an appropriate level of complexity compared to the reference.",
        "syntax": "Sentence structure in the candidate should exhibit a level of complexity similar to the reference."
    },
    "specificity": {
        "use_of_examples_and_review": "The candidate should incorporate specific examples or evidence when reflected in the reference.",
        "detailed_descriptions": "Quantitative or detailed information present in the reference should be echoed in the candidate."
    },
    "interestingness": {
        "engagement": "The candidate should maintain engagement levels comparable to the reference.",
        "kindness": "The candidate should reflect a tone of consideration or sensitivity similar to the reference.",
        "originality": "The candidate may include new perspectives, but they should be coherent with the ideas expressed in the reference."
    },
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
        return pd.read_csv("./data/sample_zero_shot.csv")

    def generate(self, criterion="linguistic_fluency", method="reference"):
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
                        output_file.write("Reference: " + row["sentence_A"] + "")
                        output_file.write("Candidate: " + row["sentence_B"] + "")
                        for sub, checklist in row["checklists"].items():
                            output_file.write(f"Criterion: {sub}")
                            output_file.write(checklist + "")
                        output_file.write("Overall Evaluation:")
                        for crit, score in row["overall_eval"].items():
                            output_file.write(f"  {crit}: {score}")
                        output_file.write("" + "="*40 + "")

                pbar.update(1)
            except Exception as e:
                print(f"Error processing row {i}: {e}")
                failed_ids.append(i)
                print(f"Failed at row index {i} with error: {e}")
                pbar.update(1)
                continue
                
       

        print("\nFinal Evaluation:")
        flat_scores = [score for scores in pred_scores.values() for score in scores]
        print("Average Score:", round(np.mean(flat_scores) * 100, 2))

        summary = {
            "average_scores": {sub: round(np.mean(pred_scores[sub]) * 100, 2) for sub in sub_criteria},
            "overall_average_score": round(np.mean(flat_scores) * 100, 2)
        }

        with open(f"results/stj_zero_{criterion}_{method}_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        if failed_ids:
            with open(f"results/stj_zero_{criterion}_{method}_failed_ids.txt", "w") as f:
                f.write("Failed IDs:\n")
                for failed_id in failed_ids:
                    f.write(f"{failed_id}\n")


        print("\nSummary saved to:", f"results/stj_zero_{criterion}_{method}_summary.json")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--method", type=str, choices=["reference", "candidate", "criterion"], default="reference")
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

        summary_path = f"results/stj_zero_{criterion}_{args.method}_summary.json"
        if os.path.exists(summary_path):
            with open(summary_path, "r") as f:
                summary = json.load(f)
                summary_scores[criterion] = summary["overall_average_score"]

    with open("results/overall_summary_zero.txt", "w") as summary_file:
        summary_file.write("Final Summary of Overall Average Scores:")
        for criterion, avg_score in summary_scores.items():
            summary_file.write(f"{criterion}: {avg_score}%")

    print("Completed evaluations. Summary written to results/overall_summary_zero.txt")
