# Import tranformer classes
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer  # poem_gen_model
from transformers import BlipProcessor, BlipForConditionalGeneration        # image_caption
from diffusers import DiffusionPipeline                                     # image_gen
from transformers import BitsAndBytesConfig                                 # optimization
from langchain.llms import HuggingFacePipeline                              # poem_gen_model
from langchain.prompts import PromptTemplate                                # Prompt template
from langchain.chains import LLMChain                                       # Helper functions
from transformers import pipeline                                           # Helper functions
# Import torch
import torch
# Import PIL's Image
from PIL import Image
# App dev framework
import streamlit as st

@st.cache_resource
def get_chain():
    # model name
    name = "meta-llama/Llama-2-7b-chat-hf"
    # vision model name
    model_image = "Salesforce/blip-image-captioning-large"
    # vision generator model name
    vision_gen = "openskyml/midjourney-mini"
    # Set auth variable from hugging face
    auth_token = ""

    # Create quantization config
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )

    # Load Image and decoder Feature extractor
    feature_extractor = BlipProcessor.from_pretrained(
        model_image,
        cache_dir="./model/",
        load_in_4bit=True,
        quantization_config=quantization_config,
    )

    # Load Image Captioning model
    img_caption_model = BlipForConditionalGeneration.from_pretrained(
        model_image,
        cache_dir="./model/",
        torch_dtype=torch.float16, 
        load_in_4bit=True,
        quantization_config=quantization_config,
    )

    # Load Image Generating model
    vision_gen_model = DiffusionPipeline.from_pretrained(
        vision_gen,
        cache_dir="./model/",
    ).to("cuda")

    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        name,
        cache_dir="./model/",
        use_auth_token=auth_token,
    )

    # Create model
    model = AutoModelForCausalLM.from_pretrained(
        name,
        cache_dir="./model/",
        use_auth_token=auth_token,
        rope_scaling={"type":"dynamic","factor":2},
        load_in_4bit=True,
        quantization_config=quantization_config,
    )

    # Create llm pipeline
    pipe = pipeline("text-generation",model=model,tokenizer=tokenizer,max_new_tokens=1024)
    llm = HuggingFacePipeline(pipeline=pipe)

    # Create poem prompt template
    prompt_template_poem = PromptTemplate(
        input_variables=["question"],
        template="""<s>[INST] <<SYS>>
    You are a helpful, respectful and honest assistant. Always answer as 
    helpfully as possible, while being safe. Your answers should not include
    any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content.
    Please ensure that your responses are socially unbiased and positive in nature. 

    Your goal is to only compose poems and reply in poetic speech. Don't reply if the request is not relate to composing a poem.<</SYS>>

    HUMAN: {question}

    ASSISTANT:[/INST]""")

    # Create story prompt template
    prompt_template_story = PromptTemplate(
        input_variables=["question"],
        template="""<s>[INST] <<SYS>>
    You are a helpful, respectful and honest assistant. Always answer as 
    helpfully as possible, while being safe. Your answers should not include
    any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content.
    Please ensure that your responses are socially unbiased and positive in nature. 

    Your goal is to only write a story. Don't reply if the request is not relate to writing a story.<</SYS>>

    HUMAN: {question}

    ASSISTANT:[/INST]""")

    # Poem LLMChain
    poem_chain = LLMChain(prompt=prompt_template_poem,llm=llm)
    # Story LLMChain
    story_chain = LLMChain(prompt=prompt_template_story,llm=llm)

    return poem_chain,story_chain,feature_extractor,img_caption_model,vision_gen_model

# Obtain chains
poem_chain,story_chain,feature_extractor,img_caption_model,vision_gen_model = get_chain()
# Create Image caption model
def Image_caption_pipe(image,text="",feature_extractor=feature_extractor,img_caption_model=img_caption_model):
    # Generate response
    inputs = feature_extractor(image,text,return_tensors="pt").to(img_caption_model.device)
    image_caption = img_caption_model.generate(**inputs,max_new_tokens=20)
    image_caption_text = feature_extractor.decode(image_caption[0],skip_special_tokens=True)
    return image_caption_text

# Generate response
def generate_prompt(image,prompt,chain_choice):
    # If image is not null then caption it
    caption = ""
    if image:
        caption = Image_caption_pipe(Image.open(image))
    # Pick choice
    if chain_choice=="Poem-Generation":
        caption = "Compose a poem based on "+caption
    elif chain_choice=="Story-Generation":
        caption = "Write a story based on "+caption
    # Generate prompt
    if image and prompt:
        prompt=caption+" and "+prompt
    else:
        prompt=caption+prompt
    return prompt



# Title
st.title("📖Quill-Bot")
# Poem or story
chain_choice = st.radio("Pick an option",["Poem-Generation","Story-Generation"])
# Prompt text box
prompt = st.text_input("Enter your prompt here...")
image = st.file_uploader("Upload the image here...",type=["png","jpg","jpeg"])
button = st.button("Generate")
response = None

# If we hit enter
if button:
    # Pass the prompt to the LLMChain
    print("Prompt under process...")
    prompt = generate_prompt(image,prompt,chain_choice)
    print(prompt)
    if chain_choice=="Poem-Generation":
        response = poem_chain.run(prompt)
    elif chain_choice=="Story-Generation":
        response = story_chain.run(prompt)
    response = response.replace("\n","  \n ")
    print("Prompt processed...Success")
    # Print response
    st.write(response)
    # Generate image
    gen_img = vision_gen_model(prompt=response).images[0]
    st.image(gen_img)
    print("Query handled")

