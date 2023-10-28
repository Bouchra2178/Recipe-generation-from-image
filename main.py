##pip install transforms torch

import os
import openai
from dotenv import find_dotenv, load_dotenv
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from transformers import pipeline
from transformers import AutoProcessor, MusicgenForConditionalGeneration
from langchain.chains import LLMChain
from PIL import Image
import streamlit as st
import requests
 #find variables env
load_dotenv(find_dotenv())
openai.api_key = os.getenv("OPENAI_API_KEY")
hugginface_api=os.getenv("HUGFACE_HUB_API_KEY")

##step 1: image to text hugging face
def image_to_text2(image):
    image = Image.open(image)  # Open the uploaded image
    image = image.convert("RGB")  # Convert the image to RGB format
    image_path = "temp_image.jpg"  # Define a temporary file to save the image

    # Save the image in a local path
    image.save(image_path, "JPEG")

    # Use a pipeline as a high-level helper
    pipe = pipeline("image-to-text", 
                    model="Salesforce/blip-image-captioning-large",
                    max_new_tokens=1000
    )
    text = pipe(image_path)[0]['generated_text']  # Process the temporary image file
    return text  # Return the extracted text

    
    

##step2: use LLM to generate recipe
#open ai chat api
model = "gpt-3.5-turbo"

chat = ChatOpenAI(temperature=0.7, model=model)
def generate_recipe(ingredients):
    template = """
    You are a extremely knowledgeable nutritionist, bodybuilder and chef who also knows
                everything one needs to know about the best quick, healthy recipes. 
                You know all there is to know about healthy foods, healthy recipes that keep 
                people lean and help them build muscles, and lose stubborn fat.
                
                You've also trained many top performers athletes in body building, and in extremely 
                amazing physique. 
                
                You understand how to help people who don't have much time and or 
                ingredients to make meals fast depending on what they can find in the kitchen. 
                Your job is to assist users with questions related to finding the best recipes and 
                cooking instructions depending on the following variables:
                0/ {ingredients}
                
                When finding the best recipes and instructions to cook,
                you'll answer with confidence and to the point.
                Keep in mind the time constraint of 5-10 minutes when coming up
                with recipes and instructions as well as the recipe.
                
                If the {ingredients} are less than 3, feel free to add a few more
                as long as they will compliment the healthy meal.
                
            
                Make sure to format your answer as follows:
                - The name of the meal as bold title (new line)
                - Best for recipe category (bold)
                    
                - Preparation Time (header)
                    
                - Difficulty (bold):
                    Easy
                - Ingredients (bold)
                    List all ingredients 
                - Kitchen tools needed (bold)
                    List kitchen tools needed
                - Instructions (bold)
                    List all instructions to put the meal together
                - Macros (bold): 
                    Total calories
                    List each ingredient calories
                    List all macros 
                    
                    Please make sure to be brief and to the point.  
                    Make the instructions easy to follow and step-by-step.
    """
    prompt=PromptTemplate(input_variables=["ingredients"],template=template)
    recipe_chain = LLMChain(llm=chat, prompt=prompt, verbose=True)
    recipe = recipe_chain.run(ingredients)
    return recipe





##step3 : text to speech
def text_to_speech(text):
    API_URL = (
        "https://api-inference.huggingface.co/models/facebook/fastspeech2-en-ljspeech"
    )
    headers = {"Authorization": f"Bearer {hugginface_api}"}

    payload = {
        "inputs": text,
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.content




def main2():
    caption = image_to_text2(image="ingredients.jpeg")
    print(caption)
    '''audio = text_to_speech(text=caption)
    with open("audio.flac", "wb") as file:
       file.write(audio)'''
    recipe = generate_recipe(ingredients=caption)
    print(recipe)
    pass




    # Define the Streamlit app
def main3():
    st.title("Image to Recipe App")
    st.header("Upload an image and get a recipe")
    # Image upload widget
    file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if file is not None:
        # Save the uploaded image to the filesystem
        with open(file.name, "wb") as f:
            f.write(file.getvalue())

        st.image(
            file,
            caption="The uploaded image",
            use_column_width=True,
            width=250
        )

               # Convert the image to text
        caption = image_to_text2(file.name)  # Pass the file directly to image_to_text()
        print(file.name)
        if caption:  # Check if a caption was extracted successfully
            # Generate recipe based on the image caption
         with st.spinner("wait few minutes .."):
            recipe = generate_recipe(ingredients=caption)

            # Display ingredients and recipe in expanders
            with st.expander("Ingredients"):
                st.write(caption)

            with st.expander("Recipe"):
                st.write(recipe)
            audio = text_to_speech(recipe)
            with open("audio.mpeg", "wb") as file:
                file.write(audio)
            with st.expander("Audio"):
                st.audio("audio.mpeg")

           
        else:
            st.write("Text extraction failed. Try uploading a different image.")

    pass


#invoking main function
if __name__=="__main__":
  main3()