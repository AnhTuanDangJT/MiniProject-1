"""
main.py
-------
Main chatbot loop for the Student Support AI.
Connects semantic search and sentiment analysis together.
"""

# Import regular expression library to help split multiple questions
import re

# Import the semantic search class from semantic_search.py
from semantic_search import SemanticSearch

# Import the sentiment analyzer class from sentiment.py
from sentiment import SentimentAnalyzer


# This function splits one user input into separate questions
def split_questions(user_input):
    # Split the text after each question mark followed by a space
    questions = re.split(r'(?<=[?])\s+', user_input)

    # Remove extra spaces and ignore empty parts
    return [question.strip() for question in questions if question.strip()]


# Main function that runs the chatbot
def main():
    # Tell the user that models are loading
    print("Loading models, please wait...\n")

    # Create the semantic search engine and load the knowledge base CSV file
    search_engine = SemanticSearch("knowledgebase.csv")

    # Create the sentiment analyzer
    sentiment_analyzer = SentimentAnalyzer()

    # Print a welcome message
    print("=" * 40)

    # Print the chatbot name
    print("  Welcome to Student Support AI")

    # Tell the user how to exit the chatbot
    print("  Type 'quit' to exit.")

    # Print the bottom border
    print("=" * 40 + "\n")

    # Create a list to store conversation history
    conversation_history = []

    # Start the main conversation loop
    while True:
        # Ask the user for input and remove extra spaces
        user_input = input("You: ").strip()

        # Check if the user wants to quit
        if user_input.lower() == "quit":
            # Print goodbye message
            print("Goodbye! Have a great day.")

            # Stop the loop
            break

        # Check if the user typed nothing
        if not user_input:
            # Ask the user to type a question
            print("Please type a question.\n")

            # Skip the rest and go back to the start of the loop
            continue

        # Split the user input in case there are multiple questions in one line
        questions = split_questions(user_input)

        # Answer each question separately
        for question in questions:
            # Analyze the sentiment of this question
            label, score = sentiment_analyzer.analyze(question)

            # Find the best answer using semantic search
            answer, similarity_score = search_engine.get_best_answer(question)

            # Print the sentiment label and confidence score
            print(f"Sentiment: {label} ({score:.2f})")

            # Check if the user seems very negative or frustrated
            if sentiment_analyzer.needs_escalation(label, score):
                # Recommend talking to a human advisor
                print("Recommended escalation: Contact human advisor.")

            # Print the answer from the knowledge base
            if similarity_score < 0.35:
                if label == "POSITIVE":
                    print(
                        "Answer: Thank you! I am glad I could help. Please ask a student support question if you need more help.\n")
                else:
                    print(
                        "Answer: I am not sure about that. Please ask a student support question or contact a human advisor.\n")
            else:
                print(f"Answer: {answer}\n")

            # Save this question and answer in conversation history
            conversation_history.append({
                # Store the user's question
                "user": question,

                # Store the sentiment label and score
                "sentiment": f"{label} ({score:.2f})",

                # Store the chatbot's answer
                "answer": answer,

                # Store the semantic similarity score for internal tracking
                "similarity_score": f"{similarity_score:.2f}"
            })


# Run the main function only when this file is executed directly
if __name__ == "__main__":
    # Start the chatbot
    main()

