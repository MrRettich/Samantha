# Samantha

Samantha is a Python application that integrates with OpenAI's API to perform a variety of tasks, including audio recording, image capturing, audio transcription, and interaction with an AI assistant. It demonstrates using the assistants API and message threads. It is intended to be used with the `GPT-4o` model of Open AI.

## Features

- **Audio Recording**: Records audio for a specified duration and saves it as a WAV file.
- **Image Capturing**: Captures images using a webcam.
- **Audio Transcription**: Transcribes an audio file using OpenAI's Whisper.
- **Image Processing**: Converts an image file to a format that OpenAI's API can process.
- **AI Interaction**: Interacts with an AI assistant, adding text and images to a thread and streaming the assistant's responses.
- **TTS**: Converts the response of the AI into speech and plays it back.
- **Message Thread**: Conversation is in one thread to enshure continuity of context.

## Dependencies

- Python 3.6+
- OpenAI Python client
- cv2
- pyaudio
- simpleaudio
- wave
- python-dotenv

You can install these dependencies using pip: `pip install -r requirements.txt`

## Setup

1. Clone the repository.
2. Install the required dependencies.
3. Create a `.env` file in the root directory and add your OpenAI API key as `OPENAI_API_KEY` and the assistant ID as `ASSIStANT_ID`.

## Usage

Run the `samantha.py` script. The script will prompt you to press Enter to start recording audio and capturing images. It will then transcribe the audio, process the images, and send them to an AI assistant. The assistant's responses will be streamed and spoken out loud.

## Functions

- `record_audio(duration)`: Records audio for a specified duration.
- `capture_image(num_pictures, delay)`: Captures a specified number of images with a specified delay between each capture.
- `transcribe_audio(file_path)`: Transcribes an audio file.
- `make_image_file(image_path)`: Converts an image file to a format that OpenAI's API can process.
- `add_image_to_thread(thread_id, file_id)`: Adds an image to a thread.
- `add_text_to_thread(thread_id, text)`: Adds text to a thread.
- `run_thread_and_stream(thread_id, assistant_id)`: Runs a thread and streams the assistant's responses.
- `text_to_speech(speech_text)`: Converts text to speech and saves it as a WAV file.
- `play_wav(file_path)`: Plays a WAV file.

## Note

This application is a demonstration of OpenAI's capabilities and should not be used for production purposes.
