from datasets import load_dataset
from config import Config
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()


class QuestionGenerator:
    def __init__(self, n_titles: int = 100):
        self.n_titles = n_titles
        self.pattern = r"List of|disambiguation"
        self.data = load_dataset(Config.dataset_name, Config.subset_name, split="train")
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.df = None

    def _sample_titles(self, seed: int, n: int) -> pd.DataFrame:
        titles = list(self.data.shuffle(seed=seed).select(range(n))["title"])
        return pd.DataFrame({"Title": titles})

    def _build_clean_titles(self):
        self.df = self._sample_titles(seed=42, n=self.n_titles)

        while True:
            # Drop titles where the articles are a list
            self.df = self.df[~self.df["Title"].str.contains(
                self.pattern, case=False, na=False
            )].reset_index(drop=True)

            # enough clean titles
            if len(self.df) >= self.n_titles:
                self.df = self.df.iloc[: self.n_titles]
                break

            # need more titles
            n_needed = self.n_titles - len(self.df)
            new_df = self._sample_titles(seed=40, n=n_needed)
            self.df = pd.concat([self.df, new_df], ignore_index=True)

    def _generate_questions(self):
        questions = []
        instructions = """
        You generate paraphrased questions in English by STRICTLY following the instructions below.

        For each topic:
        - All questions MUST ask for the same underlying information.
        - ONLY wording should change for ALL the questions (same question type, same answer).
        - Each question should be something you'd expect an average person to ask.
        - Each question must be a simple sentence, no connectors like "and", "but", "so".
        - Output ONLY a valid Python list of 3 strings, nothing else.
        """

        for topic in self.df["Title"]:
            response = self.client.responses.create(
                model="gpt-5.1",
                instructions=instructions,
                input=(
                    f"Generate 3 similar one-line questions on the topic {topic} "
                    "that you'd expect an average person to ask."
                ),
            )
            questions.append(response.output_text.replace("\n", " ").strip())

        self.df["Questions"] = questions

    def run(self, output_path: str = "list_of_questions.csv"):
        self._build_clean_titles()
        self._generate_questions()
        self.df.to_csv(output_path, index=False)
        print("DONE!")


if __name__ == "__main__":
    gen = QuestionGenerator(n_titles=100)
    gen.run()
