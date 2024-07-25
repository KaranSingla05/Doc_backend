from dotenv import load_dotenv,dotenv_values
import os
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
print(api_key)