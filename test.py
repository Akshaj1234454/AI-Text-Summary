
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

model_id = "SalmanFaroz/Llama-2-7b-samsum"
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb, device_map="auto")

prompt = f"""Summarize the following conversation:

### Input:

22/07/25, 12:05 am - Pranav: Jisko bhi chahiye wo uss bot ko apne whatapp group m add kr skta h
22/07/25, 12:05 am - Takatak: Joote itne pdenge na
22/07/25, 12:06 am - Akshaj: Ek phone number dedicate nahi karna padega?
22/07/25, 12:06 am - Pranav: Phir usko bolo ki @bot summarise last 200 messages
22/07/25, 12:06 am - Akshaj: Telegram ki tarah👍
22/07/25, 12:06 am - Pranav: Twilio ki api dekh smjh aajayega
22/07/25, 12:06 am - Akshaj: Accha Abe sexy hai phir to
22/07/25, 12:06 am - Akshaj: Ab banega project
22/07/25, 12:06 am - +91 92054 03476: Bnao bhai bano
22/07/25, 12:07 am - Pranav: Chatgpt ke saath brainstorm kr phir bana
22/07/25, 12:07 am - Pranav: Fastapi ya flask use krio
22/07/25, 12:07 am - Akshaj: Chat gpt ka hi api ghusaunga mast👍
22/07/25, 12:07 am - Akshaj: Abe bot ke behaviour ke liye iski kya jaroorat
22/07/25, 12:07 am - +91 92054 03476: Oye fastapi ki industry mein itni demand nhi hai na?
22/07/25, 12:07 am - +91 92054 03476: Ya hai demand
22/07/25, 12:07 am - Akshaj: Normal django me hi code likh dunga uska
22/07/25, 12:07 am - Akshaj: Bot ka
22/07/25, 12:08 am - Pranav: Api route aur webhooks toh lagenge, fastapi m easy pdega
22/07/25, 12:08 am - Akshaj: Bas api use hi to karna bawal thodi hai
22/07/25, 12:08 am - Pranav: Industry m sb use hota h
22/07/25, 12:08 am - Akshaj: Accha vo sikhna padega phir
22/07/25, 12:08 am - +91 92054 03476: Ok
22/07/25, 12:08 am - +91 92054 03476: Kal toh test match
22/07/25, 12:09 am - Takatak: Jldi bnade kl heman ko jarurat pdegi
22/07/25, 12:09 am - +91 92054 03476: Sahi baat hai
22/07/25, 12:09 am - +91 92054 03476: Uska koi bharosa bhi nhi ki padhne hi Beth jaaye
22/07/25, 12:10 am - Takatak: Vo padhta hi h
22/07/25, 12:10 am - +91 92054 03476: Wahi toh
22/07/25, 6:11 am - Googi: Kya bhai kya jarurat padege muje
22/07/25, 6:11 am - Googi: Short mein bata do upar 411 msg hai kon padhega itne sare ko
22/07/25, 6:12 am - Googi: ?? 🤔🤔🤔
22/07/25, 8:15 am - Takatak: Ek app ki jisse tu ye 411 message padh sake
22/07/25, 8:15 am - Takatak: Kuch nhi aise hi bakchodi chal rhi thi
22/07/25, 9:15 am - Googi: Toh banau bhai kya kar rahe ho
22/07/25, 9:16 am - Googi: Intern kar raha hai ek app nahi bana sakta
22/07/25, 9:16 am - Takatak: abe main kaha se, akshaj aur anni bhai bna rhe the
22/07/25, 9:16 am - Googi: Oye yeh bta
22/07/25, 9:16 am - Googi: Ke
22/07/25, 9:17 am - Googi: Fees ko ek sath bhar de sare phir yeh warden aise kyu bol raha hai ke hostel fee recipt
22/07/25, 9:17 am - Googi: Leke aane hau
22/07/25, 9:18 am - Takatak: fees toh sara hi bharna hoga na iss sem ka
22/07/25, 9:18 am - Googi: Ha par woh bol raha hai ke hostel lete time hostel fee recipt bhi hone chahiye
22/07/25, 9:18 am - Googi: Aur yeh batao ke kisi ne roommate wala bhara hai
22/07/25, 9:19 am - Takatak: hn toh ek print nikalwa lio yaar, dekha toh pichli baar bhi nhi tha
22/07/25, 9:19 am - Takatak: abhi dekha tha, bhar dete h
22/07/25, 9:20 am - Googi: Pichle bar alag se pay ke the hostel ke
22/07/25, 9:20 am - Googi: Iss bar toh sare ek sath pay ke hai
22/07/25, 9:20 am - Takatak: pichli baar bhi sath hi hui thi
22/07/25, 9:20 am - Googi: Alag se 30,000 nahi bhare the
22/07/25, 9:21 am - Takatak: abe vo toh kuch registration vagera ka tha na
22/07/25, 9:21 am - Googi: Acha chod mein bhi 27 ko he aa raha hu toh jo bhi chahiye hoga bata diyo wahi se print karwa lunga mein
22/07/25, 9:21 am - Takatak: ye bhi thik h
22/07/25, 9:21 am - Googi: Ha bus yeh mat bolyo ke muje bhi nahi pata
22/07/25, 9:22 am - Googi: 😂
22/07/25, 9:22 am - Takatak: arre
22/07/25, 9:22 am - Takatak: photo rkhlio bs 2
22/07/25, 9:22 am - Googi: Ha woh rakh lunga
22/07/25, 9:47 am - +91 92054 03476: This message was deleted
22/07/25, 9:53 am - +91 92054 03476: Oye sarthak jo abhi tak roommates nhi mile kya?
22/07/25, 9:54 am - Takatak: kyu kya hua
22/07/25, 9:54 am - +91 92054 03476: Uska merko message aaya hai  mai kiske saath room le rha hun
22/07/25, 9:54 am - Takatak: nhi mile honge hoskta h
22/07/25, 9:55 am - +91 92054 03476: Ansh vansh recommended karde tu
22/07/25, 9:55 am - Takatak: dost ka bura kyu hi sochna h
22/07/25, 9:55 am - +91 92054 03476: Yeh bhi hai
22/07/25, 9:55 am - +91 92054 03476: Woh gadha tab nhi bola jab mai dhund rha tha roomates
22/07/25, 9:57 am - Takatak: usne dhund toh rkhe the, pta nhi kya hua
22/07/25, 9:58 am - +91 92054 03476: Akshaj ne change karna tha na
22/07/25, 9:58 am - +91 92054 03476: Ek room mate
22/07/25, 12:34 pm - Akshaj: <Media omitted>
22/07/25, 12:34 pm - Akshaj: Oye ye bharna hai?
22/07/25, 12:34 pm - Akshaj: Register karna hai na in site se microsoft form ke pehle?
22/07/25, 12:58 pm - Googi: Ha
22/07/25, 1:08 pm - Akshaj: Abe kya tatti site hai
22/07/25, 1:08 pm - Akshaj: State select hi nahi ho raha
22/07/25, 1:08 pm - Akshaj: 😭
22/07/25, 1:09 pm - Akshaj: Swayam local chapter me yes karu ya  no
22/07/25, 1:12 pm - Akshaj: Oye
22/07/25, 1:12 pm - Akshaj: You deleted this message
22/07/25, 1:15 pm - Akshaj: Ho gaya


### Summary:"""

inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
output_ids = model.generate(inputs["input_ids"], max_new_tokens=150, do_sample=False)
summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)
