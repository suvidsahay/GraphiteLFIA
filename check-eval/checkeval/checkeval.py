from openai import OpenAI
import os, json
from instructor import OpenAISchema
from typing import List
from pydantic import BaseModel, Field, conint, field_validator,confloat
from pathlib import Path

default_criteria_definitions = {
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

class ChecklistItem(BaseModel):
    number: conint(ge=1) = Field(..., description="The item number")
    text: str = Field(..., description="The text of the checklist item")
    # importance: confloat(ge=0, le=1) = Field(1, description="The importance of the item in the checklist")
    

class Checklist(OpenAISchema):
    items: List[ChecklistItem] = Field(..., description="The checklist items")

    def to_markdown(self):
        markdown = "# Checklist\n"
        for i,item in enumerate(self.items):
            markdown += f"{i+1}: {item.text}\n"
        return markdown

class ChecklistResponseItem(BaseModel):
    item: conint(ge=1) = Field(..., description="Identifier for the checklist item.")
    # explanation: str = Field(None, description="A brief explanation of why the candidate did or did not contemplate the item.")
    isChecked: bool = Field(..., description="Indicates if the candidate contemplates the item.")
    
class ChecklistResponse(OpenAISchema):
    """The responses from the evaluation checklist."""
    items: List[ChecklistResponseItem] = Field(..., description="List of individual checklist item responses.")

    def call(self):
        results = []
        for item in self.items:
            results.append({
                "item": item.item,
                "contemplated":item.isChecked,
                # "reason":item.reason,
            })
        return results
    
    def score(self):
        return sum([item.isChecked for item in self.items])/len(self.items)

class Checkeval:
    def __init__(self, api_key, model="gpt-4o-mini", criteria_definitions=None):
        self.model = model
        self.client = OpenAI(
            # This is the default and can be omitted
            api_key=api_key,
        )
        if criteria_definitions is not None:
            self.criteria_definitions = criteria_definitions
        else:
            self.criteria_definitions = default_criterion_definitions

    def call_model(self, messages, tools=None, tool_choice=None):
        return self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            tools=tools,
            tool_choice=tool_choice,
            seed=10
        )
    
    def generate_checklist(self, criterion, text=None,  prompt=None, criterion_definition=None):
        if prompt is None:
            # prompts is a directory containing the prompt files it is in the same level of this file inside the package
            if prompt is None:
                path = Path(__file__).parent / "prompts" / "generate_checklist.md"
                prompt = open(path).read()

            if criterion_definition is None:
                criterion_definition = self.criteria_definitions.get(criterion, "No definition provided.")

            prompt = prompt.format(criterion=criterion, criterion_definition=criterion_definition)
        
        tools = [{"type":"function","function":Checklist.openai_schema}]

        messages=[
            {"role": "system", "content": prompt}
        ]

        if text is not None:
            messages.append({"role": "user", "content": f"### Reference text\n{text}"})

        chat_completion = self.call_model(messages,tools, {"type":"function","function":{"name":"Checklist"}})
        try:
            return Checklist.from_response(chat_completion)
        except Exception as e:
            print(chat_completion.choices[0].message)
            messages.append({"role":"assistant", "tool_calls":chat_completion.choices[0].message.tool_calls})
            messages.append({
                "tool_call_id": chat_completion.choices[0].message.tool_calls[0].id,
                "role": "tool",
                "name": "Checklist",
                "content": str(e),
            })
            # messages.append({"role":"user", "content": str(e)})
            chat_completion = self.call_model(messages,tools, {"type":"function","function":{"name":"Checklist"}})
            return Checklist.from_response(chat_completion)
    
    def evaluate_checklist(self, text, checklist, criterion, prompt=None,criterion_definition=None):
        if prompt is None:
            # prompts is a directory containing the prompt files it is in the same level of this file inside the package
            if prompt is None:
                path = Path(__file__).parent / "prompts" / "evaluate_checklist.md"
                prompt = open(path).read()

            if criterion_definition is None:
                criterion_definition = self.criteria_definitions.get(criterion, "No definition provided.")

            if type(checklist) != str:
                checklist = checklist.to_markdown()

            prompt = prompt.format(criterion=criterion, criterion_definition=criterion_definition, checklist=checklist)
        
        tools = [{"type":"function","function":ChecklistResponse.openai_schema}]

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"### Candidate text\n{text}"},
        ]

        chat_completion = self.call_model(messages, tools, {"type":"function","function":{"name":"ChecklistResponse"}})

        return ChecklistResponse.from_response(chat_completion)

    def reference_guided(self, criterion, reference, candidate, checklist=None,prompt=None, criterion_definition=None):
        
        if checklist is None:
            checklist = self.generate_checklist(criterion,reference, prompt=prompt)
        
        results = self.evaluate_checklist(candidate, checklist, criterion,prompt=prompt)

        return {
            "checklist":checklist,
            "results":results,
        }
        

    def candidate_guided(self, criterion, reference, candidate, checklist=None):
        return self.reference_guided(criterion, candidate, reference, checklist)

    def criterion_guided(self, criterion, reference, candidate, checklist=None):
        prompt = open(Path(__file__).parent / "prompts" / "criterion_generate.md").read()
        criterion_definition = self.criteria_definitions.get(criterion, "No definition provided.")
        if checklist is None:
            prompt = prompt.format(criterion=criterion, criterion_definition=criterion_definition)

            checklist = self.generate_checklist(reference, criterion, prompt)

        if type(checklist) != str:
            checklist = checklist.to_markdown()
        
        evaluation_prompt = open(Path(__file__).parent / "prompts" / "criterion_evaluate.md").read()

        evaluation_prompt = evaluation_prompt.format(criterion=criterion, criterion_definition=criterion_definition, checklist=checklist)

        messages = [
            {"role": "system", "content": evaluation_prompt},
            {"role": "user", "content": f"### Reference text\n{reference}\n\n### Candidate text\n{candidate}"},
        ]

        tools = [{"type":"function","function":ChecklistResponse.openai_schema}]

        chat_completion = self.call_model(messages, tools, {"type":"function","function":{"name":"ChecklistResponse"}})

        return {
            "checklist":checklist,
            "results":ChecklistResponse.from_response(chat_completion),
        }