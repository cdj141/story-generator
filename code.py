# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 04:45:25 2025

@author: 陈东杰
"""

import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.feature_extraction.text import TfidfVectorizer

# 设置 Hugging Face 缓存路径为 D:\3\cc
os.environ["HF_HOME"] = "D:/3/cc"

# Load GPT-J model
model_name = "EleutherAI/gpt-j-6B"
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="D:/3/cc")

model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    cache_dir="D:/3/cc", 
    trust_remote_code=True, 
    ignore_mismatched_sizes=True
)

# Generate text function
def generate_paragraph(prompt, max_length):
    """
    Generates a paragraph of text based on the given prompt.
    
    Args:
        prompt (str): The input prompt for the model.
        max_length (int): The maximum number of tokens to generate.

    Returns:
        str: The generated text.
    """
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_length,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.2,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Dynamic context truncation
def truncate_context(context, max_tokens=1500):
    tokens = tokenizer.encode(context)
    if len(tokens) > max_tokens:
        truncated_tokens = tokens[-max_tokens:]  # Keep the most recent tokens
        return tokenizer.decode(truncated_tokens)
    return context

# Save the story to a file
def save_story(story):
    with open("generated_story.txt", "w", encoding="utf-8") as f:
        f.write(story)
    print("Story saved to 'generated_story.txt'.")

# Select emotion for the chapter
def select_emotion():
    print("\nSelect the emotion for this chapter:")
    emotions = ["Happy", "Sad", "Exciting", "Scary", "Mysterious"]
    for i, emotion in enumerate(emotions, 1):
        print(f"{i}. {emotion}")
    try:
        emotion_choice = int(input("Enter the number corresponding to your choice: "))
        if 1 <= emotion_choice <= len(emotions):
            return emotions[emotion_choice - 1]
        else:
            print("Invalid choice, defaulting to 'Neutral'.")
            return "Neutral"
    except ValueError:
        print("Invalid input, defaulting to 'Neutral'.")
        return "Neutral"

# Main program
def main(): 
    print("Welcome to the Collaborative Story Generator!\n")
    print("Please select a story background:")
    backgrounds = [
        "A medieval European kingdom with knights, castles, and battles.",
        "An ancient Chinese empire with palaces, emperors, and martial arts.",
        "A futuristic metropolis with towering skyscrapers and advanced technology.",
        "An enchanted forest filled with magical creatures and hidden secrets.",
        "A post-apocalyptic wasteland with survivors fighting for resources."
    ]
    for i, bg in enumerate(backgrounds, 1):
        print(f"{i}. {bg}")
    print("0. Custom background")
    
    try:
        background_choice = int(input("Enter the number: "))
    except ValueError:
        print("Invalid input. Please enter a number.")
        return

    if background_choice == 0:
        print("Please describe your custom background in detail.")
        location = input("Where does the story take place? ")
        time_period = input("What is the time period (e.g., medieval, future)? ")
        main_theme = input("What is the main theme (e.g., adventure, mystery)? ")
        background = f"A story set in {location} during the {time_period} with a focus on {main_theme}."
    elif 1 <= background_choice <= len(backgrounds):
        background = backgrounds[background_choice - 1]
    else:
        print("Invalid choice. Please restart the program and select a valid option.")
        return

    print("\nSelect the story type:")
    print("1. Short story (3 chapters)")
    print("2. Medium story (5 chapters)")
    print("3. Long story (8 chapters)")
    
    try:
        story_type = int(input("\nEnter the number: "))
    except ValueError:
        print("Invalid input. Please enter a number.")
        return

    if story_type == 1:
        total_chapters = 3
        max_words_per_chapter = int(input("Enter the maximum words per chapter (default 200): ") or 200)
    elif story_type == 2:
        total_chapters = 5
        max_words_per_chapter = int(input("Enter the maximum words per chapter (default 300): ") or 300)
    elif story_type == 3:
        total_chapters = 8
        max_words_per_chapter = int(input("Enter the maximum words per chapter (default 400): ") or 400)
    else:
        print("Invalid choice. Please restart the program and select a valid option.")
        return

    # Initialize variables
    chapter_number = 1
    user_contributions = []
    full_story = f"Background: {background}\n\n"

    while chapter_number <= total_chapters:
        print(f"\n--- Chapter {chapter_number} ---")
        # Emotion selection
        chapter_emotion = select_emotion()
        
        print("You can write a part of the story or leave it empty to let the system generate it.")
        user_input = input("\nEnter your story input (or press Enter to skip): ")

        # Combine user input and generated story into context
        if user_input.strip():
            user_contributions.append(user_input.strip())
            chapter_prompt = f"\n--- Chapter {chapter_number} ---\nUser Input:\n{user_input}\n"
        else:
            chapter_prompt = f"\n--- Chapter {chapter_number} ---\n"
        
        # Always include the initial background in the prompt
        prompt = (
            f"Background: {background}\n"
            f"Emotion: {chapter_emotion}\n"
            f"Reminder: Ensure the story aligns with the theme of '{background}'.\n"
            f"User Input: {user_input.strip() if user_input.strip() else 'No input provided.'}\n"
            f"Continue the story based on the previous chapters and the above input:\n{chapter_prompt}"
        )

        print("\nGenerating story, please wait...\n")
        generated_story = generate_paragraph(prompt, max_length=max_words_per_chapter)
        if len(generated_story.split()) > max_words_per_chapter:
            generated_story = " ".join(generated_story.split()[:max_words_per_chapter])
        print("\n--- Generated Story ---\n", generated_story)

        # Update full story
        full_story += f"\n--- Chapter {chapter_number} ---\n{generated_story}\n"

        # Check if this is the final chapter
        if chapter_number == total_chapters:
            print("\n--- Generating the Ending ---\n")
            ending_prompt = (
                f"Background: {background}\n"
                f"Emotion: {chapter_emotion}\n"
                f"Reminder: Conclude the story in a satisfying way, aligning with the theme of '{background}'.\n"
                f"Final Chapter: Wrap up all loose ends and provide a strong conclusion to the story.\n"
            )
            story_ending = generate_paragraph(ending_prompt, max_length=max_words_per_chapter)
            print("\n--- Story Ending ---\n", story_ending)
            full_story += f"\n--- Story Ending ---\n{story_ending}\n"

        chapter_number += 1

    # Add summary of user contributions
    if user_contributions:
        full_story += "\n\n--- User Contributions ---\n" + "\n".join(user_contributions)

    # Save the story to a file
    save_story(full_story)

    print("\nThe story has concluded. Thank you for collaborating!")

if __name__ == "__main__":
    main()


