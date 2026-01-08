import os
from google import genai
from datetime import datetime

def generate():
    """
    Chain-of-Thought (CoT) prompting: Step-by-step reasoning process
    """
    client = genai.Client(
        api_key="AIzaSyDcrcWL6avAggXQiVC5CI-5dhSeNlt8JoY"
    )
    
    input_question = "what is datascience"
    
    # Chain-of-Thought prompt: Explicit reasoning steps
    prompt = f"""You are an IT support team member. Draft an email response using step-by-step reasoning.

User Question: {input_question}

Think through this step by step:

Step 1: Understand the user's question
- The user is asking about: [Analyze what they want to know]

Step 2: Identify the key concepts to explain
- Main topic: [Identify main topic]
- Related concepts: [List related concepts]
- User's context: [Consider why they might need this]

Step 3: Structure the response logically
- Start with a clear definition
- Break down into key components
- Provide practical examples if applicable
- Include next steps or resources

Step 4: Format as a professional email
- Subject line should be clear and informative
- Greeting: Dear User
- Body: Well-structured explanation
- Closing: Professional and helpful

Step 5: Finalize the email with proper formatting

Now, based on this reasoning process, draft the complete email:
From: it@edukron.com
To: Dear User"""
    
    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=prompt
    )

    print("=" * 50)
    print("CHAIN-OF-THOUGHT RESPONSE:")
    print("=" * 50)
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    filename = f"{output_dir}/Chain_of_thought{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

    with open(filename, "w", encoding="utf-8") as f:
        f.write(response.text)

    print(f"Output saved to {filename}")

if __name__ == "__main__":
    generate()
