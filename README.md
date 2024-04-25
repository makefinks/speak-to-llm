# speak-to-llm

![demo](/images/demo.gif)

## Requirements


### Ollama
The application uses Ollamas API for serving the local Large Language Models.

To install Ollama visit: https://ollama.com/download


### OpenAI
If you want to use Text to Speech you need to set your OpenAI API Key in as a environment variable.

```bash
export OPENAI_API_KEY = your_openai_api_key
```
If you are not familiar with the process you can visit this [guide](https://www.immersivelimit.com/tutorials/adding-your-openai-api-key-to-system-environment-variables)

## Installation

Clone the repository and install the requirements.
```bash
git clone https://github.com/makefinks/speak-to-llm.git
cd speak-to-llm
pip install -r requirements.txt
```

If you have an NVIDIA GPU it is highly recommended to install the cuda version of torch. Without this whisper will run on your CPU and will be significantly slower.

<b>check your cuda version </b>
```bash
nvcc --version
```
![version](images/image.png)

<b>Install torch (change the last part to match your version) </b>
```
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121
```

## Usage

```bash
python talk_llm.py
```

<b> Arguments </b>

> --whisper <model_name>

Which whisper model to use for transcription ([All Models](https://github.com/openai/whisper)). Use large-v3 for the best quality if you have ~10GB VRAM.

> --llm_model <model_name>

Which LLM model to run with ollama. [Here](https://ollama.com/library) is a list of available models on ollama. Defaults to Llama3-8b.


