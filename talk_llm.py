import argparse
import os
import re
import sounddevice as sd
import numpy as np
import tempfile
import soundfile as sf
import whisper
import threading
import keyboard
from rich.console import Console
from rich.spinner import Spinner
from rich.markdown import Markdown
from elevenlabs.client import ElevenLabs
from elevenlabs import stream
import ollama
from openai import OpenAI
import pyaudio
import queue

from audio import AudioManager

def preload_ollama(llm_model, console):
    with console.status(f"[bold green]Loading LLM: {llm_model}...") as status:
        model_list = ollama.list()

        if llm_model not in model_list.keys():
            ollama.pull(llm_model)
        
        ollama.chat(model=llm_model, messages=[{"role": "user", "content": "reply with 'yes'"}])

def init_console():
    console = Console()
    console.clear()
    return console

def load_whisper(model_name, console):
    with console.status("[bold green]Loading Whisper model...") as status:
        model = whisper.load_model(model_name)
    return model


def stream_llm_response(llm_model, messages, speech_queue, silent_flag, console):

    stream = ollama.chat(
        model = llm_model,
        messages=messages,
        stream=True
    )

    full_text = ""
    current_text = ""
    for chunk in stream:

        chunk_content = chunk['message']['content']
        full_text += chunk_content
        current_text += chunk_content
        console.print(chunk['message']['content'], end="")
    
        lines = re.split(r'(?<!\.\.)\.(?!\.)|(?<=[!?])\s+', current_text)
    
        if not silent_flag:
            for i, line in enumerate(lines):
                if i < len(lines) - 1:
                    # Check if it's a list item or an incomplete sentence
                    if not re.match(r"^\d+\.", line.strip()):
                        speech_queue.put(line.strip())
                    else:
                        current_text = line
                else:
                    # Last element, might be incomplete
                    current_text = line

    print("\n")
    if not silent_flag:  
        # Process any remaining text after streaming
        if current_text.strip():
            speech_queue.put(current_text.strip())

    # Append the entire response to messages once complete
    messages.append({
        "role": "assistant",
        "content": full_text
    })

def transcribe_audio(file_path, model, messages, console):
    if file_path:
        with console.status("[bold green]transcribing...") as status:
            result = model.transcribe(file_path)
            transcript = result['text']  
            console.print("[green]User: [yellow]" + result['text'])

            messages.append({
                "role": "user",
                "content": transcript
             })
    else:
        console.print("[red]No audio file provided for transcription.")

    os.remove(file_path)
    return result['text']

def main_loop():
    argparser = argparse.ArgumentParser()

    argparser.add_argument("--whisper", type=str, default="small", help="The name of the Whisper model to use.")
    argparser.add_argument("--llm_model", type=str, default="llama3", help="The name of the LLM model to use.")
    argparser.add_argument("--silent", action="store_true", help="Disable TTS.")
    argparser.add_argument("--tts", choices=["openai", "elevenlabs"], default="openai", help="The TTS service to use.")

    args = argparser.parse_args()
    messages = []
    console = Console()

    console.clear()
    model = load_whisper(model_name=args.whisper, console=console)
    client = OpenAI()

    if args.tts == "elevenlabs":
        eleven_client = ElevenLabs()

    speech_queue = queue.Queue()
    preload_ollama(llm_model=args.llm_model, console=console)

    audio_manager = AudioManager(speech_queue, console, args.tts)
    tts_thread = audio_manager.start_tts_thread(eleven_client if args.tts == "elevenlabs" else client)

    while True:
        file_path = audio_manager.record_audio()
        if file_path:
            transcript = transcribe_audio(file_path, model, messages, console)
            stream_llm_response(args.llm_model, messages, speech_queue, args.silent, console)

if __name__ == "__main__":
    main_loop()