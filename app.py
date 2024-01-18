from embed_document import embed_doc
from langchain_community.vectorstores import Chroma
import os
import warnings
import dotenv
from process_query import get_response
from langchain_openai import OpenAIEmbeddings


def new_book():
    book_name = input("Please enter the name of the book: ")
    file_name = input("Please enter the file name: ")
    file_name = './' + file_name
    book_dir = "dbs/" + book_name
    db_dir = os.path.join(os.getcwd(), book_dir)
    os.makedirs(db_dir)
    print("Please wait while this document is processed.")
    embed_doc(file_name, db_dir)
    return db_dir

def old_book():
    db_dir = input("Please enter the name of the book: ")
    db_dir = "./dbs/" + db_dir
    return db_dir


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    dotenv.load_dotenv()

    print("Welcome back to Book Club! If at any time you'd like to leave, enter 'exit'.")
    while True:
        new_or_old = input("Would you like to discuss a new book (1) or one that we have discussed before (2)? ")
        if new_or_old == "1":
            db_dir = new_book()
        elif new_or_old == "2":
            db_dir = old_book()
        else:
            break

        conversation_history = ["The following is the conversation history"]
        db = Chroma(persist_directory=db_dir, embedding_function=OpenAIEmbeddings())
        while True:
            print()
            query = input("What would you like to say? ")
            if query == "exit" or query == "Exit":
                break
            response = get_response(db, query, conversation_history)
            conversation_history.append("User: " + query)
            conversation_history.append("AI: " + response)
            print(response)
            
    print("Thanks for taking part in Book Club! Hope to see you again soon!")