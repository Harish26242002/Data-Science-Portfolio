import os
from google import genai
from datetime import datetime

def generate():
    """
    Few-shot prompting: A few examples to guide the model's response pattern
    """
    client = genai.Client(
        api_key="AIzaSyDcrcWL6avAggXQiVC5CI-5dhSeNlt8JoY"
    )
    
    input_question = "what is datascience"
    
    # Few-shot prompt: A few examples demonstrating the pattern
    prompt = f"""You are an IT support team member. Here are a few examples of how to structure email responses:

Example:
Question: What is cloud computing?
Response:
From: it@edukron.com
To: Dear User
Subject: Understanding Cloud Computing

Dear User,

Cloud computing refers to the delivery of computing services—including servers, storage, databases, networking, software, analytics, and intelligence—over the Internet ("the cloud") to offer faster innovation, flexible resources, and economies of scale.

Key benefits include:
- Cost efficiency
- Scalability
- Accessibility
- Automatic updates

If you have more questions, feel free to reach out.

Best regards,
IT Support Team
it@edukron.com

---

Now respond to this question following the same format:
Question: {input_question}

Remember:
- From: it@edukron.com
- To: Dear User
- Provide step-by-step information where applicable"""
    
    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=prompt
    )

    print("=" * 50)
    print("FEW-SHOT RESPONSE:")
    print("=" * 50)
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    filename = f"{output_dir}/few_shot{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

    with open(filename, "w", encoding="utf-8") as f:
        f.write(response.text)

    print(f"Output saved to {filename}")


if __name__ == "__main__":
    generate()
