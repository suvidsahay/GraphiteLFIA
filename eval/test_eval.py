import os
from .combined_eval import CombinedEvaluator


def read_file(filepath):
    with open(filepath, "r", encoding="utf-8") as file:
        return file.read().strip()


def main():
    evaluator = CombinedEvaluator()

    file_pairs = [
        ("eval/test/longwriter_article1.txt", "data/FreshWiki/txt/2023_Tour_de_France.txt"),
        ("eval/test/storm_article1.txt", "data/FreshWiki/txt/2023_Tour_de_France.txt"),
        ("eval/test/suri_article1.txt", "data/FreshWiki/txt/2023_Tour_de_France.txt"),
        ("eval/test/longwriter_article2.txt", "data/FreshWiki/txt/2023_Tour_de_France.txt"),
        ("eval/test/storm_article2.txt", "data/FreshWiki/txt/2023_Tour_de_France.txt"),
        ("eval/test/suri_article2.txt", "data/FreshWiki/txt/2023_Tour_de_France.txt"),
    ]

    for i, (response_file, reference_file) in enumerate(file_pairs, start=1):
        response = read_file(response_file)
        reference = read_file(reference_file)

        instruction = "Write a 2000-words article on '2023 Tour de France' highlighting the key events, riders, and the significance of the race."
        results = evaluator.evaluate_all(instruction, response, reference)

        print(f"Results for Pair {i}:")
        for category, result in results.items():
            print(f"{category}: {result}")
        print("-" * 50)


if __name__ == "__main__":
    main()
