import os
import streamlit as st
from openai import OpenAI
import tempfile
import pandas as pd
from audiorecorder import audiorecorder
from pydub import AudioSegment
from pydub.utils import which

AudioSegment.converter = which("ffmpeg")
AudioSegment.ffprobe   = which("ffprobe")

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

@st.cache_data(show_spinner=False)
def detect_language(text: str) -> str:
    """Detect whether the text is Japanese (ja) or Vietnamese (vi). Cached for speed."""
    prompt = f"Detect if the following text is Japanese or Vietnamese. Respond only with 'ja' or 'vi'.\n\nText:\n{text}"
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a language detection tool."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0,
    )
    lang = response.choices[0].message.content.strip().lower()
    return "ja" if "ja" in lang else "vi"


def translate_text(text: str, source_lang: str, target_lang: str, continue_mode: bool) -> str:
    """Translate text precisely between Japanese and Vietnamese, with optional conversation context."""
    base_prompt = (
        f"You are a professional translator. Translate the following text from "
        f"{'Japanese' if source_lang == 'ja' else 'Vietnamese'} to "
        f"{'Vietnamese' if target_lang == 'vi' else 'Japanese'}.\n"
        "The translation must be precise, natural, and without errors."
    )

    messages = [{"role": "system", "content": base_prompt}]

    if continue_mode and st.session_state.conversation:
        messages.extend(st.session_state.conversation)

    messages.append({"role": "user", "content": text})

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.0,
    )
    translation = response.choices[0].message.content.strip()

    if continue_mode:
        st.session_state.conversation.append({"role": "user", "content": text})
        st.session_state.conversation.append({"role": "assistant", "content": translation})

    return translation


def speech_to_text(audio_file) -> str:
    """Convert speech to text using Whisper."""
    with open(audio_file, "rb") as f:
        transcript = client.audio.transcriptions.create(
            model="gpt-4o-mini-transcribe",
            file=f
        )
    return transcript.text.strip()


@st.cache_data(show_spinner=False)
def text_to_speech_cached(text: str) -> bytes:
    """Convert text to speech and cache the audio result for reuse."""
    voice = "alloy"
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        audio_file_path = tmp.name
        response = client.audio.speech.create(
            model="gpt-4o-mini-tts",
            voice=voice,
            input=text
        )
        response.stream_to_file(audio_file_path)
    with open(audio_file_path, "rb") as f:
        audio_bytes = f.read()
    return audio_bytes


# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="JA ⇆ VI Translator Chat", layout="centered")

st.title("🌏 Japanese ⇆ Vietnamese Translator (Cached Audio Flags)")
st.write("Type or speak in Japanese or Vietnamese. Auto-detects language, remembers context, and lets you click flags 🇯🇵 🇻🇳 to play cached audio instantly.")

# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = []
if "conversation" not in st.session_state:
    st.session_state.conversation = []

# Continue conversation toggle
continue_mode = st.checkbox("🔄 Continue Conversation Mode", value=True)

tab1, tab2 = st.tabs(["✍️ Text Input", "🎤 Live Microphone Input"])

# ----- TEXT MODE -----
with tab1:
    with st.form("text_translation_form"):
        text_input = st.text_area("Enter your text (Japanese or Vietnamese):", height=150)
        submit_text = st.form_submit_button("Translate")

    if submit_text and text_input.strip():
        with st.spinner("Detecting language (cached)..."):
            source_lang = detect_language(text_input)
            target_lang = "vi" if source_lang == "ja" else "ja"

        with st.spinner("Translating..."):
            try:
                translation = translate_text(text_input, source_lang, target_lang, continue_mode)
                st.success(f"✅ Translation complete! ({'JA→VI' if source_lang=='ja' else 'VI→JA'})")
                st.text_area("Translated text:", translation, height=150)

                # Save to history
                st.session_state.history.append({
                    "input": text_input,
                    "translation": translation,
                    "direction": f"{'JA→VI' if source_lang=='ja' else 'VI→JA'}"
                })

            except Exception as e:
                st.error(f"Error: {e}")

# ----- LIVE MICROPHONE MODE -----
with tab2:
    st.info("Click to record your voice, then stop to process.")
    audio = audiorecorder("🎤 Start Recording", "⏹ Stop Recording")

    if len(audio) > 0:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            audio.export(tmp.name, format="wav")
            audio_path = tmp.name

        with st.spinner("Transcribing speech..."):
            try:
                transcribed_text = speech_to_text(audio_path)

                with st.spinner("Detecting language (cached)..."):
                    source_lang = detect_language(transcribed_text)
                    target_lang = "vi" if source_lang == "ja" else "ja"

                translation = translate_text(transcribed_text, source_lang, target_lang, continue_mode)

                st.success(f"✅ Done! ({'JA→VI' if source_lang=='ja' else 'VI→JA'})")
                st.write(f"**Recognized Speech:** {transcribed_text}")
                st.text_area("Translated Text:", translation, height=150)

                # Save to history
                st.session_state.history.append({
                    "input": transcribed_text,
                    "translation": translation,
                    "direction": f"{'JA→VI' if source_lang=='ja' else 'VI→JA'}"
                })

            except Exception as e:
                st.error(f"Error: {e}")


# ----- Conversation History -----
st.subheader("🗒️ Conversation History (Bilingual Chat with Cached Audio Flags)")

# Buttons for managing history
col1, col2, col3 = st.columns([1,1,2])

with col1:
    if st.session_state.history and st.button("🗑️ Clear History"):
        st.session_state.history.clear()
        st.session_state.conversation.clear()
        st.experimental_rerun()

with col2:
    if st.session_state.history:
        txt_content = "\n\n".join(
            [f"[{h['direction']}]\nInput: {h['input']}\nTranslation: {h['translation']}" 
             for h in st.session_state.history]
        )
        st.download_button(
            "⬇️ Download TXT",
            data=txt_content,
            file_name="translation_history.txt",
            mime="text/plain"
        )

with col3:
    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        csv_content = df.to_csv(index=False)
        st.download_button(
            "⬇️ Download CSV",
            data=csv_content,
            file_name="translation_history.csv",
            mime="text/csv"
        )

# Bilingual Chat Layout with Cached Audio Flags
if st.session_state.history:
    for i, entry in enumerate(reversed(st.session_state.history)):
        col_left, col_right = st.columns(2)
        src_flag = "🇯🇵" if entry['direction'].startswith("JA") else "🇻🇳"
        tgt_flag = "🇻🇳" if src_flag == "🇯🇵" else "🇯🇵"

        with col_left:
            st.markdown(f"{src_flag} **Input**")
            st.info(entry['input'])
            if st.button(f"{src_flag} 🔊 Play", key=f"play_src_{i}"):
                st.audio(text_to_speech_cached(entry['input']), format="audio/mp3")

        with col_right:
            st.markdown(f"{tgt_flag} **Translation**")
            st.success(entry['translation'])
            if st.button(f"{tgt_flag} 🔊 Play", key=f"play_tgt_{i}"):
                st.audio(text_to_speech_cached(entry['translation']), format="audio/mp3")

        st.divider()
else:
    st.info("No conversation history yet. Start translating!")
while ran the above code, the error show as below
FileNotFoundError: This app has encountered an error. The original error message is redacted to prevent data leaks. Full error details have been recorded in the logs (if you're on Streamlit Cloud, click on 'Manage app' in the lower right of your app).
Traceback:
File "/mount/src/ja-vi-translator/app.py", line 136, in <module>
    audio = audiorecorder("🎤 Start Recording", "⏹ Stop Recording")
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/home/adminuser/venv/lib/python3.11/site-packages/audiorecorder/__init__.py", line 51, in audiorecorder
    audio_segment = AudioSegment.from_file(BytesIO(b64decode(base64_audio)))
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/home/adminuser/venv/lib/python3.11/site-packages/pydub/audio_segment.py", line 728, in from_file
    info = mediainfo_json(orig_file, read_ahead_limit=read_ahead_limit)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/home/adminuser/venv/lib/python3.11/site-packages/pydub/utils.py", line 274, in mediainfo_json
    res = Popen(command, stdin=stdin_parameter, stdout=PIPE, stderr=PIPE)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/usr/local/lib/python3.11/subprocess.py", line 1026, in __init__
    self._execute_child(args, executable, preexec_fn, close_fds,
File "/usr/local/lib/python3.11/subprocess.py", line 1955, in _execute_child
    raise child_exception_type(errno_num, err_msg, err_filename)
