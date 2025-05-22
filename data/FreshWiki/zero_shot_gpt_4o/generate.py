from openai import OpenAI
import os
import time

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

def generate_outline(topic):
    prompt = (
        f"You are an expert writer. Given a prompt, generate a detailed and structured outline "
        f"with headings and subheadings for a 2000-word article.\n\n"
        f"Prompt: Write a 2000 word article on \"{topic}\""
    )

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    print(response)
    return response.choices[0].message.content

def generate_article(outline):
    prompt = (
        "Using the following outline, write a detailed 2000-word article. "
        "Expand each section with relevant information, examples, and explanations.\n\n"
        f"Outline:\n{outline}"
    )

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    print(response)
    return response.choices[0].message.content

def process_files(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.endswith(".txt"):
            topic = filename[:-4].replace("_", " ")  # Remove .txt and replace _ with space
            output_path = os.path.join(output_dir, filename)

            if os.path.exists(output_path):
                print(f"Skipping (already exists): {filename}")
                continue

            print(f"\nProcessing: {topic}")

            try:
                outline = generate_outline(topic)
                article = generate_article(outline)

                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(article)
                    print(f"Article saved to: {output_path}")
            except Exception as e:
                print(f"Failed to generate article for {topic}: {e}")

def main():
    input_dir = "data/FreshWiki/txt"
    output_dir = "data/zero_shot_gpt_4o"

    process_files(input_dir, output_dir)

if __name__ == "__main__":
    main()
