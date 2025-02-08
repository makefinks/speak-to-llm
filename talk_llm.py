import argparse
import os
import re
import sys
import torch
from rich.console import Console
from elevenlabs.client import ElevenLabs
import ollama
from openai import OpenAI
import queue
from faster_whisper import WhisperModel
from audio import AudioManager


def preload_ollama(llm_model, console):
    with console.status(f"[bold green]Loading LLM: {llm_model}...") as status:
        available_models = [model.model.split(":")[0] for model in ollama.list().models]
        if llm_model not in available_models:
            ollama.pull(llm_model)

        ollama.chat(
            model=llm_model, messages=[{"role": "user", "content": "reply with 'yes'"}]
        )


def init_console():
    console = Console()
    console.clear()
    return console


def stream_llm_response(llm_model, messages, speech_queue, silent_flag, console):
    stream = ollama.chat(model=llm_model, messages=messages, stream=True)

    full_text = ""
    current_text = ""
    for chunk in stream:
        chunk_content = chunk["message"]["content"]
        full_text += chunk_content
        current_text += chunk_content
        console.print(chunk["message"]["content"], end="")

        lines = re.split(r"(?<!\.\.)\.(?!\.)|(?<=[!?])\s+", current_text)

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
    messages.append({"role": "assistant", "content": full_text})


def transcribe_audio(file_path, model: WhisperModel, messages, console):
    if not file_path:
        console.print("[red]No audio file provided for transcription.")
        return ""
    try:
        with console.status("[bold green]transcribing..."):
            segments, info = model.transcribe(file_path)
            transcription = " ".join(segment.text for segment in segments)
            console.print("[green]User: [yellow]" + transcription)
            messages.append({"role": "user", "content": transcription})
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)
    return transcription

def load_whisper(model_name, console):
    # check if cuda is available
    if not torch.cuda.is_available():
        console.print("[red]CUDA is not available. Using CPU for inference.")
        model = WhisperModel(model_name, device="gpu", compute_type="float32")
    else:
        model = WhisperModel(model_name, device="cuda", compute_type="float16")
    return model


def main_loop():
    argparser = argparse.ArgumentParser()

    argparser.add_argument(
        "--whisper",
        type=str,
        default="small",
        help="The name of the Whisper model to use.",
    )
    argparser.add_argument(
        "--llm_model",
        type=str,
        default="qwen:7b",
        help="The name of the LLM model to use.",
    )
    argparser.add_argument("--silent", action="store_true", help="Disable TTS.")
    argparser.add_argument(
        "--tts",
        choices=["openai", "elevenlabs"],
        default="openai",
        help="The TTS service to use.",
    )
    argparser.add_argument(
        "--lang",
        type=str,
        choices=["en", "multi"],
        default="en",
        help="The language to use for the TTS service.",
    )
    argparser.add_argument(
        "--voice_id",
        type=str,
        default="ErXwobaYiN019PkySvjV",
        help="The voice ID to use for Eleven Labs TTS.",
    )

    args = argparser.parse_args()
    messages = []
    console = Console()
    console.clear()

    # Load the whisper model
    model = load_whisper(args.whisper, console)

    client = OpenAI()
    if args.tts == "elevenlabs":
        eleven_client = ElevenLabs()

    speech_queue = queue.Queue()
    preload_ollama(llm_model=args.llm_model, console=console)

    audio_manager = AudioManager(
        speech_queue, console, args.tts, args.lang, args.voice_id
    )

    tts_thread = audio_manager.start_tts_thread(
        eleven_client if args.tts == "elevenlabs" else client
    )

    # Main loop
    try:
        while True:
            file_path = audio_manager.record_audio()
            if file_path:
                transcript = transcribe_audio(file_path, model, messages, console)
                stream_llm_response(
                    args.llm_model, messages, speech_queue, args.silent, console
                )
    except KeyboardInterrupt:
        console.print("\n[red]Exiting application...[/red]")
        os._exit(0)

if __name__ == "__main__":
    main_loop()

