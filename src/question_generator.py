import os
import re

def generate_questions(text):
    """Generates questions from the text of a single file."""
    title_match = re.search(r"Title: (.+)", text)
    title = title_match.group(1).strip() if title_match else "Unknown Disease"

    keywords = re.findall(r"\n([A-Z][a-zA-Z ]+?)\n", text)

    questions = [f"What is the {keyword} for {title}?" for keyword in keywords]

    return questions

def process_folder(folder_path):
    """Processes all text files in a folder and generates questions."""
    all_questions = {}

    # Iterate through all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):  # Process only .txt files
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read()
                questions = generate_questions(text)
                all_questions[filename] = questions

    return all_questions

# Example usage
if __name__ == "__main__":
    folder_path = "./articles/W"  # Replace with your folder path
    questions_dict = process_folder(folder_path)
    print(questions_dict)
    print("***************************")
    # Print questions for each file
    for filename, questions in questions_dict.items():

        print(f"\nQuestions for {filename}:")
        for question in questions:
            print(f"- {question}")