# pip install google-genai

import os
from google import genai
from datetime import datetime

def generate():
    client = genai.Client(
        api_key="AIzaSyDcrcWL6avAggXQiVC5CI-5dhSeNlt8JoY"
    )
    input_question = (
    "I am unable to log in to the system due to an authentication issue. "
    "Could you please help resolve this problem?"
    )


    prompt = f"""
You are an IT support team member.

User Issue:
{input_question}

Instructions:
- Draft a professional email response
- This response is from the IT team
- From: it@edukron.com
- To: Dear User
- Clearly explain the resolution
- Provide step-by-step instructions specific to GitHub private repository access
- Include subject, greeting, body, and closing
- Keep the tone professional and helpful

Draft the email:
"""

    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=prompt
    )

    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    filename = f"{output_dir}/app_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

    with open(filename, "w", encoding="utf-8") as f:
     f.write(response.text)

    print(f"Output saved to {filename}")


if __name__ == "__main__":
    generate()
