
![img](https://github.com/eminbayrak/llma-chat-bot/assets/32653645/2a578537-8981-4c4d-b55c-1fa788c21501)

---

# Chat-Bot with Document Retrieval and Conversational AI

This code implements a chat-bot that utilizes document retrieval and conversational AI techniques. The chat-bot allows users to ask questions about uploaded documents, and it responds with relevant information from the documents using a combination of vector retrieval and a conversational AI model.

## Getting Started

To run the chat-bot, follow these steps:

1. Clone the repository:

```bash
git clone <repository_url>
```

2. Navigate to the project directory:

```bash
cd <project_directory>
```

3. Install dependencies using `pip`:

```bash
pip install -r requirements.txt
```

4. Download the model weights from the following link and place them in the project directory:

   - [llama-2-7b-chat.ggmlv3.q8_0.bin](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/blob/main/llama-2-7b-chat.ggmlv3.q8_0.bin)

5. Run the Streamlit app:

```bash
python -m streamlit run .\app.py
```

This will launch the chat-bot application in your web browser.

## Usage

1. Upload PDF documents:

   - Use the file uploader in the sidebar to upload PDF documents that you want the chat-bot to process.
   - Click the "Process" button to extract text and create a vector database for document retrieval.

2. Ask Questions:

   - Type your question in the chat input field.
   - Click "Send" to submit your question to the chat-bot.

3. Chat-Bot Response:
   - The chat-bot will respond with relevant information from the uploaded documents.
   - It combines document retrieval with a conversational AI model to provide accurate answers.

## Dependencies

All the required dependencies are listed in the `requirements.txt` file. You can install them using the following command:

```bash
pip install -r requirements.txt
```

## Project Structure

- `app.py`: Main application code that implements the Streamlit UI, document processing, chat-bot interaction, and conversational AI integration.
- `data/`: Directory where uploaded PDF documents are stored.
- `vectorstore/`: Directory where vector databases are stored.
- `requirements.txt`: List of required Python packages with their versions.

## Notes

- The `llama-2-7b-chat.ggmlv3.q8_0.bin` model weights should be downloaded from the provided link and placed in the project directory before running the app.

---
