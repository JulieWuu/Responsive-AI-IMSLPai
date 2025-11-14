import pandas as pd
import sqlite3
import google.generativeai as genai


# --- Step 1: Data Preparation (Done ONCE, offline) ---
# This part of the code is what you would run separately to create your database.
# For this example, let's assume you have scraped or downloaded a CSV file
# named 'imslp_data.csv' with columns like:
# 'title', 'composer', 'instrumentation', 'year', 'difficulty', 'length_minutes', 'composer_nationality', etc.

def create_database_from_csv(csv_path='raw_data.csv', db_path='imslp.db'):
    """
    Reads a CSV file with IMSLP data and saves it to a local SQLite database.
    This is a one-time setup function.
    """
    print(f"Creating database from {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
        df.columns = (df.columns.astype(str).str.strip()
                      .str.replace(r'[\s\r\n]+', ' ', regex=True))
        df.to_csv("final_data.csv", index=False)
        conn = sqlite3.connect(db_path)
        # The table will be named after the CSV file's name without the extension
        table_name = "final_data.csv".split('.')[0].replace('-', '_')  # Sanitize table name
        df.to_sql(table_name, conn, if_exists='replace', index=False)
        conn.close()
        print(f"Database '{db_path}' created successfully with table '{table_name}'.")
    except FileNotFoundError:
        print(f"Error: The file {csv_path} was not found. Please create it first.")
    except Exception as e:
        print(f"An error occurred: {e}")


# --- Step 2: The Core Application Logic (RAG) ---

class IMSLPChatbot:
    def __init__(self, db_path: str, api_key: str, system_instruction: str):
        """
        Initializes the chatbot, connects to the database, and configures the AI model.
        """
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path)
        self.system_instruction = system_instruction

        # Configure the Google Generative AI model
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash-lite')
        self.chat_space = self.model.start_chat(enable_automatic_function_calling=True)

    def chat(self):
        """
        Main chat loop to interact with the user.
        """
        print("Hello! I am IMSLPai, your personal music AI. Ask me anything about the music in my database.")
        print("Type 'quit' to exit.")

        while True:
            user_input = input("\nYou: ")
            if user_input.lower() == 'quit':
                print("IMSLPai: Goodbye!")
                break

            # RAG pipeline
            # retrieved_data = self.retrieve_relevant_info(user_input)
            # final_answer = self.generate_response(user_input, retrieved_data)

            response = self.chat_space.send_message(user_input)
            print(f"\nIMSLPai: {response.text}")

        self.conn.close()


def main():
    # --- ONE-TIME SETUP ---
    # You would run this function once to build your database from a CSV.
    # For subsequent runs, you can comment this line out.
    create_database_from_csv()

    system_prompt = ("You are a helpful AI that answers all questions related to composers and their works."
                     "The user may ask questions about a composer, in which case you shall search for that composer in your database, and provide the user with some basic information about their works and personal life."
                     "The user may ask questions about a composer and their piece for an instrument, in which case you shall look for all of the composer's work that involves that instrument, and list out a few of them."
                     "The user may ask questions about a specific piece of a composer, in which case you shall find information about that piece in your database, and also provide more detailed information such as the year the piece was composed, the average duration of that piece, the style, key and number of movements in that piece."
                     "The user may ask you to recommend some pieces of a composer, in which case you shall list out the ones that are most renowned or has the style that suits the user most."
                     "The user may ask you to compare different composers about their works, in which case you shall find all the composers mentioned and compare their data, especially the era of the composer, their work style and proportion of each instrument in their pieces."
                     "However, if the user is asking about all or many pieces at once, you shall only provide essential details about each piece, ideally under 50 words."
                     "")

    chatbot = IMSLPChatbot(db_path='imslp.db', api_key="", system_instruction=system_prompt)
    chatbot.chat()


if __name__ == '__main__':
    main()

