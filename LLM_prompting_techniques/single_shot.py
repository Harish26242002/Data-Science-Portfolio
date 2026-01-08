import os
from google import genai
from datetime import datetime


def generate():
    """
    Single-shot prompting: Direct prompt without examples
    """
    client = genai.Client(
        api_key="AIzaSyDcrcWL6avAggXQiVC5CI-5dhSeNlt8JoY"
    )
    
    input_question = "what is datascience"
    
    # Single-shot prompt: Direct instruction
    prompt = f"""You are an IT support team member. Draft a professional email response to the following user question.

User Question: {input_question}

Instructions:
- From: it@edukron.com
- To: Dear User
- Provide a clear, professional response to the user's question
- Format the email appropriately with subject line, greeting, body, and closing

Draft the email:"""
    
    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=prompt
    )

    print("=" * 50)
    print("SINGLE-SHOT RESPONSE:")
    print("=" * 50)
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    filename = f"{output_dir}/single_shot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

    with open(filename, "w", encoding="utf-8") as f:
        f.write(response.text)

    print(f"Output saved to {filename}")


if __name__ == "__main__":
    generate()
