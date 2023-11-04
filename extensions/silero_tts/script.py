import os
import html
import json
import random
import time
from pathlib import Path

import gradio as gr
import torch

from extensions.silero_tts import tts_preprocessor
from modules import chat, shared, ui_chat
from modules.utils import gradio



torch._C._jit_set_profiling_mode(False)


params = {
    'activate': False,
    'speaker': 'en_94',
    'language': 'English',
    'model_id': 'v3_en',
    'sample_rate': 48000,
    'device': 'cpu',
    'show_text': False,
    'autoplay': False,
    'voice_pitch': 'medium',
    'voice_speed': 'medium',
    'local_cache_path': ''  # User can override the default cache path to something other via settings.json
}

current_params = params.copy()

with open(Path("extensions/silero_tts/languages.json"), encoding='utf8') as f:
    languages = json.load(f)

voice_pitches = ['x-low', 'low', 'medium', 'high', 'x-high']
voice_speeds = ['x-slow', 'slow', 'medium', 'fast', 'x-fast']

# Used for making text xml compatible, needed for voice pitch and speed control
table = str.maketrans({
    "<": "&lt;",
    ">": "&gt;",
    "&": "&amp;",
    "'": "&apos;",
    '"': "&quot;",
})


def xmlesc(txt):
    return txt.translate(table)


def load_model():
    torch_cache_path = torch.hub.get_dir() if params['local_cache_path'] == '' else params['local_cache_path']
    model_path = torch_cache_path + "/snakers4_silero-models_master/src/silero/model/" + params['model_id'] + ".pt"
    if Path(model_path).is_file():
        print(f'\nUsing Silero TTS cached checkpoint found at {torch_cache_path}')
        model, example_text = torch.hub.load(repo_or_dir=torch_cache_path + '/snakers4_silero-models_master/', model='silero_tts', language=languages[params['language']]["lang_id"], speaker=params['model_id'], source='local', path=model_path, force_reload=True)
    else:
        print(f'\nSilero TTS cache not found at {torch_cache_path}. Attempting to download...')
        model, example_text = torch.hub.load(repo_or_dir='snakers4/silero-models', model='silero_tts', language=languages[params['language']]["lang_id"], speaker=params['model_id'])
    model.to(params['device'])
    return model


def remove_tts_from_history(history):
    for i, entry in enumerate(history['internal']):
        history['visible'][i] = [history['visible'][i][0], entry[1]]

    return history


def toggle_text_in_history(history):
    for i, entry in enumerate(history['visible']):
        visible_reply = entry[1]
        if visible_reply.startswith('<audio'):
            if params['show_text']:
                reply = history['internal'][i][1]
                history['visible'][i] = [history['visible'][i][0], f"{visible_reply.split('</audio>')[0]}</audio>\n\n{reply}"]
            else:
                history['visible'][i] = [history['visible'][i][0], f"{visible_reply.split('</audio>')[0]}</audio>"]

    return history


def state_modifier(state):
    if not params['activate']:
        return state

    state['stream'] = False
    return state


def input_modifier(string, state):
    if not params['activate']:
        return string

    shared.processing_message = "*Is recording a voice message...*"
    return string


def history_modifier(history):
    # Remove autoplay from the last reply
    if len(history['internal']) > 0:
        history['visible'][-1] = [
            history['visible'][-1][0],
            history['visible'][-1][1].replace('controls autoplay>', 'controls>')
        ]

    return history


def output_modifier(string, state):
    global model, current_params, streaming_state

    for i in params:
        if params[i] != current_params[i]:
            model = load_model()
            current_params = params.copy()
            break

    if not params['activate']:
        return string

    original_string = string

    string = tts_preprocessor.preprocess(html.unescape(string))

    if string == '':
        string = '*Empty reply, try regenerating*'
    else:
        output_file = Path(f'extensions/silero_tts/outputs/{state["character_menu"]}_{int(time.time())}.wav')
        prosody = '<prosody rate="{}" pitch="{}">'.format(params['voice_speed'], params['voice_pitch'])
        silero_input = f'<speak>{prosody}{xmlesc(string)}</prosody></speak>'
        model.save_wav(ssml_text=silero_input, speaker=params['speaker'], sample_rate=int(params['sample_rate']), audio_path=str(output_file))

        autoplay = 'autoplay' if params['autoplay'] else ''
        string = f'<audio src="file/{output_file.as_posix()}" controls {autoplay}></audio>'
        if params['show_text']:
            string += f'\n\n{original_string}'

    shared.processing_message = "*Is typing...*"
    return string

def generate_html_page(directory):
    # Define the file paths
    audio_files = sorted(Path(directory).rglob('*.wav'))
    text_files = sorted(Path(directory).rglob('*.txt'))

    # Extract file names
    audio_files_names = [audio_file.name for audio_file in audio_files]
    text_files_names = [text_file.name for text_file in text_files]

    # Read the content of the text files
    text_contents = []
    for text_file in text_files:
        with open(text_file, 'r') as file:
            text_contents.append(file.read().replace('"', '\"'))

    # Define the Javascript part
    javascript = f"""
    <script>
        const audioFiles = {str(audio_files_names)};
        const textContents = {str(text_contents)};

        let audioIndex = 0;
        let audioElement = new Audio(audioFiles[audioIndex]);

        const audioPlayer = document.getElementById('audio-player');
        const textDisplay = document.getElementById('text-display');
        const playButton = document.getElementById('play-button');
        const themeButton = document.getElementById('theme-button');

        audioPlayer.appendChild(audioElement);

        textDisplay.innerText = textContents[audioIndex];

        audioElement.addEventListener('ended', function () {{
            audioIndex++;
            if (audioIndex < audioFiles.length) {{
                audioElement.src = audioFiles[audioIndex];
                textDisplay.style.opacity = '0';
                setTimeout(function () {{
                    textDisplay.innerText = textContents[audioIndex];
                    textDisplay.style.opacity = '1';
                }}, 500);  // reduced transition time to 0.5 seconds
                audioElement.play();
            }} else {{  // If all audio files have been played
                audioIndex = 0;  // Reset the audio index
                audioElement.src = audioFiles[audioIndex];  // Set the audio source to the first audio file
                textDisplay.style.opacity = '0';  // Set the opacity of textDisplay to 0
                setTimeout(function () {{
                    textDisplay.innerText = textContents[audioIndex];  // Update the text display with index 0
                    textDisplay.style.opacity = '1';  // Set the opacity of textDisplay to 1
                }}, 500);
                playButton.style.display = '';  // Show the "Start Playback" button again
            }}
        }});

        playButton.addEventListener('click', function () {{
            audioElement.play();
            playButton.style.display = 'none';
        }});

        themeButton.addEventListener('click', function () {{
            document.body.classList.toggle('dark-theme');
        }});
        document.body.classList.toggle('dark-theme');
    </script>
    """

    # Generate the HTML content
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Audio Playback</title>
        <style>
            #text-display {{
                margin-top: 2em;
                font-size: 1.5em;
                opacity: 1;
                transition: opacity 0.5s;
            }}
            /* Dark theme CSS */
            .dark-theme {{
                background-color: #121212;
                color: white;
            }}
        </style>
    </head>
    <body>
        <button id="play-button">Start Playback</button>
        <button id="theme-button">Toggle Dark/Light Theme</button>
        <div id="audio-player"></div>
        <div id="text-display"></div>
        {javascript}
    </body>
    </html>
    """

    # Write the HTML content to a file
    with open(Path(directory) / 'playback.html', 'w') as file:
        file.write(html)

def process_last_reply_with_TTS(history):
    global model, current_params, streaming_state
    for i in params:
        if params[i] != current_params[i]:
            model = load_model()
            current_params = params.copy()
            break

    if len(history['visible']) == 0:
        print('No history')
        return

    # Get the last visible reply
    string = history['visible'][-1][1].replace("&#x27;","'").replace("&quot;",'"').replace("&amp;","&").replace("&lt;","<").replace("&gt;",">")

    # Create a new directory for the outputs
    current_time = int(time.time())
    output_directory = Path(f"extensions/silero_tts/outputs/{shared.settings['character']}_{current_time}")
    output_directory.mkdir(parents=True, exist_ok=True)

    paragraphs = string.split('\n')

    # Remove empty paragraphs or paragraphs that only contain whitespace
    paragraphs = [paragraph for paragraph in paragraphs if paragraph.strip() != '']

    audio_files = []
    text_files = []
    file_idx = 0

    for paragraph in paragraphs:
        preprocessed_paragraph = tts_preprocessor.preprocess(html.unescape(paragraph))

        # Pad the file index with leading zeroes
        padded_idx = str(file_idx).zfill(3)
        padded_length = str(len(paragraphs)).zfill(3)
        print(f'Generating audio file {padded_idx}/{padded_length}...')
        print(f'Paragraph: {paragraph}')

        output_file = output_directory / f'voice_{padded_idx}.wav'
        text_file = output_directory / f'paragraph_{padded_idx}.txt'

        # Save the original paragraph to a text file before generating audio
        with open(text_file, 'w') as f:
            f.write(paragraph)

        prosody = '<prosody rate="{}" pitch="{}">'.format(params['voice_speed'], params['voice_pitch'])
        silero_input = f'<speak>{prosody}{xmlesc(preprocessed_paragraph)}</prosody></speak>'
        model.save_wav(ssml_text=silero_input, speaker=params['speaker'], sample_rate=int(params['sample_rate']), audio_path=str(output_file))

        audio_files.append(str(output_file))
        text_files.append(str(text_file))

        file_idx += 1  # Increment file index only after a successful file creation

    generate_html_page(str(output_directory))  # Generate the HTML page

    # Create the output string with links to the audio files
    output_string = ''
    for idx, (audio_file, text_file) in enumerate(zip(audio_files, text_files)):
        output_string += f'<audio src="file/{audio_file}" controls></audio>'
        output_string += paragraphs[idx]
    output_string += f'<p><a href="file/{str(output_directory)}/playback.html" target="_blank">Link to HTML Playback Page</a></p>'

    # Replace the last visible reply with the output string
    history['visible'][-1] = [history['visible'][-1][0], output_string]

    return history

def setup():
    global model
    model = load_model()


def random_sentence():
    with open(Path("extensions/silero_tts/harvard_sentences.txt")) as f:
        return random.choice(list(f))


def voice_preview(string):
    global model, current_params, streaming_state

    for i in params:
        if params[i] != current_params[i]:
            model = load_model()
            current_params = params.copy()
            break

    string = tts_preprocessor.preprocess(string or random_sentence())

    output_file = Path('extensions/silero_tts/outputs/voice_preview.wav')
    prosody = f"<prosody rate=\"{params['voice_speed']}\" pitch=\"{params['voice_pitch']}\">"
    silero_input = f'<speak>{prosody}{xmlesc(string)}</prosody></speak>'
    model.save_wav(ssml_text=silero_input, speaker=params['speaker'], sample_rate=int(params['sample_rate']), audio_path=str(output_file))

    return f'<audio src="file/{output_file.as_posix()}?{int(time.time())}" controls autoplay></audio>'


def language_change(lang):
    global params
    params.update({"language": lang, "speaker": languages[lang]["default_voice"], "model_id": languages[lang]["model_id"]})
    return gr.update(choices=languages[lang]["voices"], value=languages[lang]["default_voice"])


def custom_css():
    path_to_css = Path(__file__).parent.resolve() / 'style.css'
    return open(path_to_css, 'r').read()


def ui():
    # Gradio elements
    with gr.Accordion("Silero TTS"):
        with gr.Row():
            activate = gr.Checkbox(value=params['activate'], label='Activate TTS')
            autoplay = gr.Checkbox(value=params['autoplay'], label='Play TTS automatically')

        show_text = gr.Checkbox(value=params['show_text'], label='Show message text under audio player')
        
        with gr.Row():
            language = gr.Dropdown(value=params['language'], choices=sorted(languages.keys()), label='Language')
            voice = gr.Dropdown(value=params['speaker'], choices=languages[params['language']]["voices"], label='TTS voice')
        with gr.Row():
            v_pitch = gr.Dropdown(value=params['voice_pitch'], choices=voice_pitches, label='Voice pitch')
            v_speed = gr.Dropdown(value=params['voice_speed'], choices=voice_speeds, label='Voice speed')

        with gr.Row():
            preview_text = gr.Text(show_label=False, placeholder="Preview text", elem_id="silero_preview_text")
            preview_play = gr.Button("Preview")
            preview_audio = gr.HTML(visible=False)

        with gr.Row():
            process_last_reply = gr.Button('Process last reply with TTS')
            convert = gr.Button('Permanently replace audios with the message texts')
            convert_cancel = gr.Button('Cancel', visible=False)
            convert_confirm = gr.Button('Confirm (cannot be undone)', variant="stop", visible=False)

    # Gradio callbacks
    process_last_reply.click(process_last_reply_with_TTS, gradio('history'), gradio('history')).then(
        chat.save_history, shared.gradio['mode'], None, show_progress=False).then(
        chat.redraw_html, shared.reload_inputs, shared.gradio['display'])

    # Convert history with confirmation
    convert_arr = [convert_confirm, convert, convert_cancel]
    convert.click(lambda: [gr.update(visible=True), gr.update(visible=False), gr.update(visible=True)], None, convert_arr)
    convert_confirm.click(
        lambda: [gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)], None, convert_arr).then(
        remove_tts_from_history, gradio('history'), gradio('history')).then(
        chat.save_history, gradio('history', 'unique_id', 'character_menu', 'mode'), None).then(
        chat.redraw_html, gradio(ui_chat.reload_arr), gradio('display'))

    convert_cancel.click(lambda: [gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)], None, convert_arr)

    # Toggle message text in history
    show_text.change(
        lambda x: params.update({"show_text": x}), show_text, None).then(
        toggle_text_in_history, gradio('history'), gradio('history')).then(
        chat.save_history, gradio('history', 'unique_id', 'character_menu', 'mode'), None).then(
        chat.redraw_html, gradio(ui_chat.reload_arr), gradio('display'))

    # Event functions to update the parameters in the backend
    activate.change(lambda x: params.update({"activate": x}), activate, None)
    autoplay.change(lambda x: params.update({"autoplay": x}), autoplay, None)
    language.change(language_change, language, voice, show_progress=False)
    voice.change(lambda x: params.update({"speaker": x}), voice, None)
    v_pitch.change(lambda x: params.update({"voice_pitch": x}), v_pitch, None)
    v_speed.change(lambda x: params.update({"voice_speed": x}), v_speed, None)

    # Play preview
    preview_text.submit(voice_preview, preview_text, preview_audio)
    preview_play.click(voice_preview, preview_text, preview_audio)
