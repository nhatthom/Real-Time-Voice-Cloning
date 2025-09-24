# -*- coding: utf-8 -*-
import sys, os, torch, subprocess, tempfile
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog,
    QComboBox, QMessageBox, QInputDialog
)
from pyannote.audio import Pipeline
import whisper
from transformers import MarianMTModel, MarianTokenizer
import torchaudio
import numpy as np

# ====== Real-Time Voice Cloning ======
from encoder import inference as encoder
from synthesizer.inference import Synthesizer
from vocoder import inference as vocoder
import librosa

TOKEN_FILE = ".hf_token"


class VideoTranslator(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pro Video Translator (Voice Cloning)")
        self.setGeometry(200, 200, 600, 400)

        self.input_file = None
        self.output_dir = None
        self.hf_token = self.load_hf_token()

        # Layout
        layout = QVBoxLayout()

        self.label_input = QLabel("Chưa chọn video")
        self.btn_input = QPushButton("Chọn File Video")
        self.btn_input.clicked.connect(self.load_input_video)

        self.label_output = QLabel("Chưa chọn thư mục xuất")
        self.btn_output = QPushButton("Chọn Thư Mục Xuất")
        self.btn_output.clicked.connect(self.choose_output_dir)

        self.lang_in = QComboBox()
        self.lang_out = QComboBox()
        langs = ["en", "fr", "ja", "ko", "es", "vi", "de"]
        self.lang_in.addItems(langs)
        self.lang_out.addItems(langs)

        self.btn_token = QPushButton("Nhập HF_TOKEN")
        self.btn_token.clicked.connect(self.set_hf_token)

        self.btn_process = QPushButton("Bắt đầu xử lý (Voice Clone)")
        self.btn_process.clicked.connect(self.process_video)

        layout.addWidget(QLabel("Video đầu vào:"))
        layout.addWidget(self.label_input)
        layout.addWidget(self.btn_input)

        layout.addWidget(QLabel("Thư mục xuất:"))
        layout.addWidget(self.label_output)
        layout.addWidget(self.btn_output)

        layout.addWidget(QLabel("Ngôn ngữ đầu vào:"))
        layout.addWidget(self.lang_in)
        layout.addWidget(QLabel("Ngôn ngữ đầu ra:"))
        layout.addWidget(self.lang_out)

        layout.addWidget(self.btn_token)
        layout.addWidget(self.btn_process)

        self.setLayout(layout)

    def load_hf_token(self):
        if os.path.exists(TOKEN_FILE):
            with open(TOKEN_FILE, "r", encoding="utf-8") as f:
                return f.read().strip()
        return None

    def set_hf_token(self):
        token, ok = QInputDialog.getText(self, "HF_TOKEN", "Nhập HuggingFace Token của bạn:")
        if ok and token:
            self.hf_token = token.strip()
            with open(TOKEN_FILE, "w", encoding="utf-8") as f:
                f.write(self.hf_token)
            QMessageBox.information(self, "Lưu token", "HF_TOKEN đã được lưu thành công!")

    def load_input_video(self):
        file, _ = QFileDialog.getOpenFileName(
            self, "Chọn video", "", "Video Files (*.mp4 *.mkv *.avi *.mov)"
        )
        if file:
            self.input_file = file
            self.label_input.setText(f"Đã chọn: {file}")
        else:
            self.label_input.setText("Chưa chọn video")

    def choose_output_dir(self):
        dir_ = QFileDialog.getExistingDirectory(self, "Chọn thư mục xuất")
        if dir_:
            self.output_dir = dir_
            self.label_output.setText(f"Xuất ra: {dir_}")

    def process_video(self):
        if not self.input_file or not self.output_dir:
            QMessageBox.warning(self, "Thiếu dữ liệu", "Hãy chọn video và thư mục xuất.")
            return
        if not self.hf_token:
            QMessageBox.warning(self, "Thiếu Token", "Vui lòng nhập HF_TOKEN trước khi xử lý.")
            return

        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"

            # === Step 1: Speaker diarization
            pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=self.hf_token)
            diarization = pipeline(self.input_file)

            # === Step 2: Transcribe
            whisper_model = whisper.load_model("small", device=device)
            result = whisper_model.transcribe(self.input_file, language=self.lang_in.currentText())

            # === Step 3: Translate
            in_lang = self.lang_in.currentText()
            out_lang = self.lang_out.currentText()
            model_name = f"Helsinki-NLP/opus-mt-{in_lang}-{out_lang}"
            tokenizer = MarianTokenizer.from_pretrained(model_name)
            translator = MarianMTModel.from_pretrained(model_name).to(device)

            def translate(text):
                inputs = tokenizer(text, return_tensors="pt", truncation=True).to(device)
                translated = translator.generate(**inputs)
                return tokenizer.decode(translated[0], skip_special_tokens=True)

            translated_texts = [translate(seg["text"]) for seg in result["segments"]]

            # === Step 4: Init Voice Cloning
            encoder.load_model("encoder/saved_models/pretrained.pt")
            synthesizer = Synthesizer("synthesizer/saved_models/pretrained/pretrained.pt")
            vocoder.load_model("vocoder/saved_models/pretrained/pretrained.pt")

            audio_segments = []
            with tempfile.TemporaryDirectory() as tmpdir:
                for i, seg in enumerate(result["segments"]):
                    start, end, text = seg["start"], seg["end"], translated_texts[i]

                    seg_wav = os.path.join(tmpdir, f"seg_{i}.wav")
                    subprocess.run(
                        f'ffmpeg -y -i "{self.input_file}" -ss {start} -to {end} -vn -ar 16000 "{seg_wav}"',
                        shell=True,
                    )

                    # Encode speaker embedding
                    wav, sr = librosa.load(seg_wav, sr=16000)
                    embed = encoder.embed_utterance(wav)

                    # Synthesize translated speech
                    specs = synthesizer.synthesize_spectrograms([text], [embed])
                    generated_wav = vocoder.infer_waveform(specs[0])

                    out_wav = os.path.join(self.output_dir, f"seg_{i}.wav")
                    torchaudio.save(out_wav, torch.tensor([generated_wav]), 22050)
                    audio_segments.append(out_wav)

            # === Step 5: Merge audio
            concat_list = os.path.join(self.output_dir, "list.txt")
            with open(concat_list, "w", encoding="utf-8") as f:
                for a in audio_segments:
                    f.write(f"file '{a}'\n")

            merged_audio = os.path.join(self.output_dir, "final_audio.wav")
            subprocess.run(f'ffmpeg -y -f concat -safe 0 -i "{concat_list}" -c copy "{merged_audio}"', shell=True)

            out_video = os.path.join(self.output_dir, "output.mp4")
            subprocess.run(
                f'ffmpeg -y -i "{self.input_file}" -i "{merged_audio}" -map 0:v -map 1:a -c:v copy -shortest "{out_video}"',
                shell=True,
            )

            QMessageBox.information(self, "Xong", f"Video đã xuất ra: {out_video}")

        except Exception as e:
            QMessageBox.critical(self, "Lỗi", str(e))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoTranslator()
    window.show()
    sys.exit(app.exec_())
