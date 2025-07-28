from django.shortcuts import render, redirect
from django.http import HttpResponse
from difflib import SequenceMatcher
import re

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# Load once at the top (reuse across calls)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarizeString(chat_text, min_len=50, max_len=512):
    
    if not chat_text.strip():
        return "No messages to summarize."

    word_count = len(chat_text.split())
    if word_count < min_len:
        return "Not enough content for a meaningful summary."

    try:
        result = summarizer(chat_text, min_length=min_len, max_length=max_len, do_sample=False)
        return result[0]['summary_text']
    except Exception as e:
        return f"Summarization error: {str(e)}"

def is_similar(a, b, threshold=0.6):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio() >= threshold


def group_messages(raw_lines):
    messages = []
    current_message = ""

    # WhatsApp message regex pattern (adjust if needed)
    pattern = re.compile(r"^\d{2}/\d{2}/\d{2}, \d{1,2}:\d{2}.* - ")

    for line in raw_lines:
        if pattern.match(line):
            if current_message:
                messages.append(current_message.strip())
            current_message = line
        else:
            current_message += "\n" + line

    if current_message:
        messages.append(current_message.strip())

    return messages


def home(request):
    return render(request, 'home.html')

def summarize_view(request):
    if request.method == 'POST':
        uploaded_file = request.FILES.get('textfile')
        user_prompt = request.POST.get('user_input')

        if uploaded_file and user_prompt:
            text_data = uploaded_file.read().decode('utf-8')
            # Now `text_data` is the file's content as a string
        
        else:
            return HttpResponse("""
                <script>
                    alert("Please make sure all fields are filled out.");
                    window.location.href = "/";
                </script>
            """)

        # You now have both file content and the text input
        # You can process them however you want here:
        temp = text_data.replace('\u202f', " ")
        fixedTemp = temp.splitlines()
        grouped_messages = group_messages(fixedTemp)
        print(user_prompt)
        aiInput = []
        for i in range(len(grouped_messages)-1, -1, -1):
            if is_similar(user_prompt, grouped_messages[i]):
                
                for j in range(i, len(grouped_messages)):
                    aiInput.append(grouped_messages[j])
                break
            else:
                pass
        aiInput = "\n\n".join(aiInput)
        summary = summarizeString(aiInput)
        print (summary)


        
       

        # For testing purposes:
        

    return HttpResponse("yeah wtv")