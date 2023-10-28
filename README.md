# Image to Recipe Streamlit App
The Image to Recipe Streamlit App is a versatile tool that generates recipes based on extracted image captions. Additionally, it offers users the ability to listen to the generated recipe in audio format.
## Key Functionalities
- **Image to Text Conversion:** Utilizes Hugging Face's image-to-text pipeline to extract descriptive text from uploaded images.
- **Recipe Generation using GPT-3.5:** Generates a recipe based on the extracted image caption using OpenAI's GPT-3.5 model.
- **Text-to-Speech:** Converts the generated recipe into audio for users to listen to the recipe.

## Instructions
1. **Image Upload:** Users can upload images (supported formats: jpg, jpeg, png).
2. **Image Caption:** Displays the generated text from the uploaded image.
3. **Recipe Generation:** Uses the extracted caption to generate a recipe.
4. **Audio Output:** Allows users to listen to the generated recipe as speech.

## How to Use
1. Clone the repository.
2. Install the necessary dependencies using `pip install -r requirements.txt`.
3. Run the application by executing `streamlit run main.py`.

## Technologies Used
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [OpenAI GPT-3.5](https://beta.openai.com/)
- [Streamlit](https://streamlit.io/)
