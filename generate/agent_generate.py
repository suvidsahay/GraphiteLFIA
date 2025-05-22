import os
import json
import openai
from openai import OpenAI
import time
from typing import List, Dict, Any, Optional
import re
import argparse


OPENAI_API_KEY = ""

from openai import OpenAI

client = OpenAI(api_key=OPENAI_API_KEY)

class ArticleAgent:
    """Base class for article-related agents."""

    def __init__(self, model="gpt-4o-mini"):
        self.model = model
        self.conversation_history = []

    def _call_api(self, messages: List[Dict[str, str]], temperature: float = 0.7, max_tokens: int = 4000) -> str:
        """Call the OpenAI API with retry logic."""
        retries = 3
        # print("messages - \n ", messages)
        for attempt in range(retries):
            # print("attempt - ", attempt)
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                return response.choices[0].message.content
            except Exception as e:
                if attempt < retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    print(f"API error: {e}. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    raise e

    def update_history(self, role: str, content: str, length: int = 3):
        """Add a message to the conversation history."""
        if(len(self.conversation_history) == length):
          self.conversation_history.pop(0)
        self.conversation_history.append({"role": role, "content": content})

class ArticleGenerator(ArticleAgent):
    """Agent responsible for generating article based on topic and instructions."""

    def __init__(self, model="gpt-4o-mini"):
        super().__init__(model)
        self.system_prompt = f"""You are an informational article genearator. You task is to generate a long informational article on a given topic by following the provided instructions.
        You might also be given your previous generation and certain feedbacks. If the feedbacks are given, use the feedback to refine your article."""

    def generate_article(self, topic: str, instructions: str, previous_article: Optional[str] = None,
                     feedback: Optional[str] = None) -> str:
        """Generate article based on topic, instructions and optional feedback."""
        messages = [{"role": "system", "content": self.system_prompt}]

        # Add conversation history
        messages.extend(self.conversation_history)

        # Create user prompt

        # generate boring and non-instructions article in the first try
        actual_user_prompt = f"Topic:\n{topic}\n\nInstructions:\n{instructions}\n\n"
        if(len(self.conversation_history) == 0):
          user_prompt = f"Topic:\n{topic}\n\nInstructions:\nThe article should be boring, repetitive and non-coherent.\n\n"
        else:
          user_prompt = actual_user_prompt
        if previous_article:
            user_prompt += f"Previous Aticle:\n```\n{previous_article}\n```\n\n"
        if feedback:
            user_prompt += f"Feedback to address:\n{feedback}\n\n"
        user_prompt += "Generate the article:"

        # print("user prompt - \n ", user_prompt)

        messages.append({"role": "user", "content": user_prompt})

        # Get response from API
        response = self._call_api(messages)

        # print("generation response - \n ", response)

        # Extract article from response
        article = self._extract_article(response)

        # Update conversation history
        self.update_history("user", actual_user_prompt)
        self.update_history("assistant", response)

        return article

    def _extract_article(self, response: str) -> str:
        """Extract article from the API response."""
        # Look for article blocks with ```
        if "```" in response:
            article_blocks = response.split("```")
            # Article blocks are typically in positions 1, 3, 5, etc.
            for i in range(1, len(article_blocks), 2):
                # Remove language identifier if present
                article = article_blocks[i]
                if article and not article.strip().startswith("output") and not article.strip().startswith("Output"):
                    lang_split = article.split("\n", 1)
                    if len(lang_split) > 1:
                        # Remove language identifier
                        code = lang_split[1]
                    return article.strip()



        # If no code blocks found, return the whole response
        return response


class ArticleEditor(ArticleAgent):
    """Agent responsible for analyzing article and providing feedback."""

    def __init__(self, model="gpt-4o-mini"):
        super().__init__(model)
        self.system_prompt = f"""You are an expert informational article reviewer. Your task is to review the given informational article on a specified topic and check if it follows the user-provided instructions.
        If not, then provide the feedbacks. If there are any previous feedbacks, also check if the current article follows the earlier feedbacks. If there are no improvements required, then simply respond 'Looks Good'."""

    def check_word_count(self, instructions: str, article: str) -> int:
      """Check if the article follows the specified word limit"""

      word_length = 500

      match = re.search(r'word limit:\s*(\d+)\s*words', instructions, re.IGNORECASE)
      if match:
        word_length = int(match.group(1))
      
      word_count = len(article.split())

      feedback = ""
      if(word_count < (word_length-200)):
          feedback = f"The word count is much less than the limit of {word_length} words."
      elif(word_count > (word_length+100)):
          feedback = f"The word count is much more than the limit of {word_length} words."

      return feedback

    def edit_article(self, topic: str, instructions: str, article: str) -> str:
        """Review the given article and provide feedback."""
        messages = [{"role": "system", "content": self.system_prompt}]

        # Add conversation history
        messages.extend(self.conversation_history)

        # Create user prompt
        user_prompt = f"Topic:\n{topic}\n\n"
        user_prompt += f"Instructions:\n{instructions}\n\n"
        user_prompt += f"Article:\n```\n{article}\n```\n\n"
        
        user_prompt += "Please provide detailed feedback on the article:"

        messages.append({"role": "user", "content": user_prompt})

        # Get response from API
        response = self._call_api(messages)

        length_feedback = self.check_word_count(instructions, article)

        response += "\n\n" + length_feedback

        # print("feedback - \n ", response)
        
        # Update conversation history
        self.update_history("user", user_prompt)
        self.update_history("assistant", response)

        return response

class ArticleGenEditSystem:
    """System that coordinates the article generation and editing process."""

    def __init__(self, model="gpt-4o-mini"):
        self.generator = ArticleGenerator(model)
        self.editor = ArticleEditor(model)
        self.iterations = 0
        self.max_iterations = 5

    def article_generation(self, topic: str, instructions: str) -> Dict[str, Any]:
        """Generate an article through iterative generation and editing."""
        results = {
            "topic": topic,
            "iterations": []
        }

        article = None
        feedback = None

        for i in range(self.max_iterations):
            self.iterations = i + 1
            print(f"\n--- Iteration {self.iterations} ---")

            # Generate article
            print("Generating article...")
            article = self.generator.generate_article(topic, instructions, article, feedback)

            # Review article
            print("Reviewing article...")
            feedback = self.editor.edit_article(topic, instructions, article)

            # Save iteration results
            results["iterations"].append({
                "iteration": self.iterations,
                "article": article,
                "feedback": feedback
            })

            # Check if feedback indicates the solution is correct
            if "correct" in feedback.lower() or "looks good" in feedback.lower():
                print("Article appears to be correct. Stopping iterations.")
                break

            if i == self.max_iterations - 1:
                print(f"Reached maximum number of iterations ({self.max_iterations}).")

        results["final_article"] = article
        results["iterations_count"] = self.iterations

        return results


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--topic", type=str, help="Topic of the article")
    parser.add_argument("--length", type=int, default=500, help="Word limit")
    parser.add_argument("--instructions", type=str, default="", help="Instructions for article")
    parser.add_argument("--output_file", type=str, default="agent_results.json", help="Output path for the file")

    args = parser.parse_args()
    
    topic = args.topic
    word_limit = args.length
    output_file = args.output_file
    if(args.instructions == ""):
        instructions = args.instructions
    else:
        instructions = f"""1. The article should be non-repetitive and coherent.
2. It should be clear, very engaging and interesting to read.
3. Word limit: {word_limit} words.
"""

    system = ArticleGenEditSystem()
    results = system.article_generation(topic, instructions)

    # Print final results
    print("\n=== Final Article ===")
    print(results["final_article"])

    # Save results to file
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
