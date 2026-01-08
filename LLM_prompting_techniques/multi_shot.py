import os
from google import genai
from datetime import datetime

def generate():
    """
    Multi-shot prompting: Multiple examples in the prompt to guide the model
    """
    client = genai.Client(
        api_key="AIzaSyDcrcWL6avAggXQiVC5CI-5dhSeNlt8JoY"
    )
    
    input_question = "what is datascience"
    
    # Multi-shot prompt: Multiple examples
    prompt = f"""You are an IT support team member. Draft professional email responses based on these examples:

Example 1:
User Question: How do I reset my password?
Email Response:
From: it@edukron.com
To: Dear User
Subject: Password Reset Instructions

Dear User,

Thank you for contacting the IT support team. To reset your password, please follow these steps:
1. Go to the login page
2. Click on "Forgot Password"
3. Enter your email address
4. Check your inbox for reset instructions
5. Follow the link to create a new password

If you need further assistance, please don't hesitate to contact us.

Best regards,
IT Support Team
it@edukron.com

---

Example 2:
User Question: My laptop is running slowly
Email Response:
From: it@edukron.com
To: Dear User
Subject: Troubleshooting Slow Laptop Performance

Dear User,

Thank you for reaching out. Here are some steps to improve your laptop's performance:
1. Restart your laptop
2. Close unnecessary applications
3. Clear temporary files
4. Check for available disk space
5. Run antivirus scan

If the issue persists, please contact us for further assistance.

Best regards,
IT Support Team
it@edukron.com

---

Now, draft an email response for the following user question:
User Question: {input_question}

Follow the same format and style as the examples above.
From: it@edukron.com
To: Dear User"""
    
    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=prompt
    )

    print("=" * 50)
    print("MULTI-SHOT RESPONSE:")
    print("=" * 50)
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    filename = f"{output_dir}/multi_shot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

    with open(filename, "w", encoding="utf-8") as f:
        f.write(response.text)

    print(f"Output saved to {filename}")


if __name__ == "__main__":
    generate()
