from transformers import pipeline

summarizer = pipeline("summarization", model="ai4bharat/indicBART")

def preprocess_whatsapp_chat(raw_chat):
    import re
    # Remove timestamps and phone numbers/usernames
    lines = raw_chat.strip().split('\n')
    cleaned = []
    for line in lines:
        # Remove timestamp and username
        match = re.match(r'\d{2}/\d{2}/\d{2}, \d{1,2}:\d{2}\s[ap]m - .*?: (.*)', line)
        if match:
            cleaned.append(match.group(1))
    return " ".join(cleaned)

chat_text = """
21/07/25, 11:33 pm - +91 92054 03476: Floor ka tu baat karliyo
21/07/25, 11:33 pm - Takatak: badhiya h vaise, samaan aaram se pahuch jaega
21/07/25, 11:33 pm - Pranav: Poori building paani paani hongti
21/07/25, 11:33 pm - +91 92054 03476: Hain ! Yeh kaha se news mili
21/07/25, 11:34 pm - Takatak: bennett h toh mumkin h
21/07/25, 11:34 pm - +91 92054 03476: World class facilities
21/07/25, 11:34 pm - Takatak: han uske knowledge mei dedunga
21/07/25, 11:34 pm - +91 92054 03476: Ok
21/07/25, 11:34 pm - Takatak: enrollment kya h tera
21/07/25, 11:34 pm - +91 92054 03476: 1613
21/07/25, 11:34 pm - Takatak: okay
21/07/25, 11:35 pm - Pranav: https://www.instagram.com/reel/C5aUf7BLX3w/?igsh=MThuc3Q2OW1naWQ1OQ==
21/07/25, 11:35 pm - Takatak: kaun pahucha hua h college abhi
21/07/25, 11:35 pm - Pranav: Suppli wale
21/07/25, 11:35 pm - Takatak: gaandu date toh dekh
21/07/25, 11:35 pm - Takatak: 2024 ka h
21/07/25, 11:36 pm - Pranav: Arey bc mrko abhi kisi ne bheja 😂
21/07/25, 11:36 pm - Takatak: nice bhai
21/07/25, 11:36 pm - Pranav: Anmol ka asar
"""

from transformers import pipeline

summarizer = pipeline("summarization", model="google/pegasus-xsum")

summary = summarizer(chat_text, max_length=80, min_length=20, do_sample=False)

print(summary[0]['summary_text'])