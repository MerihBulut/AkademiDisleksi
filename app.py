import json
import os
import tempfile
import threading
import unicodedata
import urllib.request
from datetime import date
from io import BytesIO
from pathlib import Path

import azure.cognitiveservices.speech as speechsdk
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from pydub import AudioSegment
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

load_dotenv()

st.set_page_config(page_title="Okuma Analiz", layout="wide")

ERROR_CATEGORIES = [
    "Fonolojik ve Ses Birleştirme Hataları",
    "Bellek ve İşlemleme Hataları",
    "Görsel-Algısal ve Dikkat Hataları",
    "Okuduğunu Anlama ve Gramer",
]

SILENCE_PAUSE_SECONDS = 1.0
PAUSE_SYMBOL = "⌛"

def load_error_rules_text() -> str:
    rules_path = Path(__file__).with_name("hata_kurallari.json")
    with rules_path.open("r", encoding="utf-8") as rules_file:
        rules = json.load(rules_file).get("kurallar", [])
    return "\n".join(
        (
            f"{rule['id']}. tanim: {rule['tanim']} "
            f"| ai_kilavuzu: {rule.get('ai_kilavuzu', '')}"
        )
        for rule in rules
    )


HATA_KURALLARI_TEXT = load_error_rules_text()


def normalize_utf8_text(text: str) -> str:
    return unicodedata.normalize("NFC", text or "")


def replace_pause_tags(text: str) -> str:
    return normalize_utf8_text(str(text or "").replace("[DURAKLAMA]", PAUSE_SYMBOL))


SYSTEM_PROMPT = f"""Sen profesyonel bir disleksi uzmanısın. Sana öğrencinin orijinal okuması gereken metni ve Azure'un BİRE BİR kaydettiği ham okuma transkriptini gönderiyorum.

GÖREVİN:
- "hata_kurallari.txt" dosyasındaki 60 kuralı temel al. Aşağıda bu dosyanın içeriği verilmiştir:
{HATA_KURALLARI_TEXT}
- Analiz sonuçlarını asla parçalara bölme. Tüm transkripti ilk saniyeden son saniyeye kadar bütüncül olarak değerlendir.
- Uzman Raporu'nu eski detaylı klinik formatta ve açıklayıcı paragraflarla yaz. Rapor içinde şu başlıkları KESİNLİKLE kullan:
  ## Akademik Beceriler
  ## Bilişsel Beceriler
  ## Sosyal, Duygusal ve Davranışsal Alan
  ## Aileye Öneriler
  ## Kısa Vadeli Hedefler
- Her başlık altında öğrencinin okumasından somut örnekler ver; yalnızca kısa özet yazma.
- Uzun transkriptler sana 40 saniyelik parçalar halinde gönderilebilir. Her parçayı bağımsız analiz et; parça dışı bağlam varsayımı yapma.
- HİBRİT MOD:
  1. Kelime Katmanı: Azure'dan gelen transkripti kelime kelime tara ve "hata_kurallari.txt" içindeki maddelerle (özellikle ID 1, 2, 3, 4, 27, 28 vb.) birebir eşleştir. En küçük kelime, harf, ses, ekleme, silme ve değiştirme hatasını bile kaçırma.
  2. Cümle Katmanı: Ardından cümlenin tamamına bakarak bağlamsal hataları (özellikle ID 32, 37, 42, 54, 56 vb.) değerlendir.
  3. Eğer bir kelime hem kelime düzeyinde (örn: heceleme) hem de cümle düzeyinde (örn: fiil değiştirme/bağlam hatası) hata içeriyorsa, rapora her iki durumu da ayrı satırlar olarak yansıt.
- Sana verilen metin bloğunu önce bütünüyle incele, sonra hatalı olduğunu düşündüğün kısımları "hata_kurallari.txt" ile karşılaştır.
- Öğrencinin okumasındaki tekrarları, hecelemeleri, duraksamaları ve ses değiştirmeleri bu 60 kural ile eşleştir.
- Transkriptteki her "⌛" duraklama sembolünü fazla duraklama olarak değerlendir ve uygun olduğunda ID 42 ile eşleştir.
- "e-e-elma" gibi durumlarda bunu 3 ayrı hata olarak değil, "elma" kelimesinin hece/ses tekrarı olarak tanımla ve uygun ID'yi ver.
- Önce Azure'dan gelen ham transkripti bütün olarak oku, sonra tüm kural eşleştirmelerini raporun bütünlüğünü bozmadan yap.
- Hataları sınıflandırırken sadece "tanim" alanına bakarak yorum yapma; her kuralın "ai_kilavuzu" alanındaki mekanik şartları da kullan.
- JSON'da tanımlı olmayan hiçbir şeyi hata olarak raporlama.
- Metinlerdeki Türkçe karakterlerin (ı, İ, ş, Ş, ğ, Ğ, ç, Ç, ö, Ö, ü, Ü) korunmasına maksimum hassasiyet göster. Özellikle kelime eşleştirmelerinde "akil" ve "aklı" gibi kelimeleri Türkçe karakter kurallarına göre analiz et.

JSON ÇIKTI KURALLARI:
Yanıtını yalnızca geçerli JSON olarak ver. Şu yapıyı kullan:
{{
  "uzman_raporu": "## Akademik Beceriler\\n\\nDetaylı klinik/eğitsel değerlendirme...\\n\\n## Bilişsel Beceriler\\n\\n...\\n\\n## Sosyal, Duygusal ve Davranışsal Alan\\n\\n...\\n\\n## Aileye Öneriler\\n\\n...\\n\\n## Kısa Vadeli Hedefler\\n\\n...",
  "hata_kategorileri": {{
    "Fonolojik ve Ses Birleştirme Hataları": 0,
    "Bellek ve İşlemleme Hataları": 0,
    "Görsel-Algısal ve Dikkat Hataları": 0,
    "Okuduğunu Anlama ve Gramer": 0
  }},
  "error_timeline": [
    {{
      "time": "00:12",
      "category": "Fonolojik ve Ses Birleştirme Hataları",
      "sub_error": "ID 17 - Kelimenin bir bölümünü tekrar etme",
      "rule_id": 17,
      "rule_definition": "Kelimenin bir bölümünü tekrar etme",
      "rule_example": "e-e-elma",
      "student_reading": "e-e-elma",
      "expected_reading": "elma",
      "description": "Öğrenci beklenen elma kelimesini e-e-elma biçiminde ses/hece tekrarıyla okudu."
    }}
  ]
}}

Her hata için en az şu bilgileri doldur: "hata_id/rule_id", "tanim/rule_definition", "gerceklesen/student_reading", "beklenen/expected_reading".
Hata tespit ederken rapora sadece HATA YAPILAN KELİMEYİ yaz. "student_reading" alanına asla 4-5 kelimelik cümle koyma. Sadece hatalı okunan tek kelimeyi yaz (örn: "yögerne"). "expected_reading" alanına da sadece beklenen doğru tek kelimeyi yaz (örn: "yönerge"). Eğer hata cümle bazlıysa cümleyi "description" alanına kısa not olarak ekle.
Kural kitabındaki genel örnek metni "rule_example" alanına yaz; bu örnek PDF'te "Madde Tanımı" içinde gösterilecektir. "description" alanına sadece öğrencinin bu testte yaptığı spesifik hatayı yaz; genel kural örneği, tanım tekrarı veya "Örnek:" ifadesi ekleme.
Tablo oluştururken "time" ve "rule_id" alanlarını kısa, doğrudan ve tek satırlık değerler olarak yaz. Bu iki alana açıklama, özel karakter, satır kırılımı veya uzun metin ekleme; Markdown tablosunda nowrap etkisi için hücre içeriklerini olduğu gibi kısa değerlerle bırak. """


def get_openai_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", None)
    if not api_key or api_key == "BURAYA_API_ANAHTARINI_YAPISTIR":
        st.error("OPENAI_API_KEY tanımlı değil. `.streamlit/secrets.toml` dosyasına anahtarınızı ekleyin.")
        st.stop()
    return OpenAI(api_key=api_key)


def get_azure_speech_credentials() -> tuple[str, str]:
    speech_key = os.environ.get("AZURE_SPEECH_KEY")
    speech_region = os.environ.get("AZURE_SPEECH_REGION")
    if not speech_key or not speech_region:
        st.error("AZURE_SPEECH_KEY ve AZURE_SPEECH_REGION `.env` dosyasında veya ortam değişkenlerinde tanımlı olmalıdır.")
        st.stop()
    return speech_key, speech_region


def ticks_to_seconds(value: int | float | None) -> float | None:
    if value is None:
        return None
    return float(value) / 10_000_000


def parse_azure_transcription_results(results: list[dict], fallback_text: str) -> tuple[str, list[dict], str]:
    transcript_parts = []
    words = []

    for result_data in results:
        nbest = result_data.get("NBest", [{}])
        best = nbest[0] if nbest else {}
        display_text = best.get("Display") or result_data.get("DisplayText") or result_data.get("Text") or ""
        if display_text:
            transcript_parts.append(normalize_utf8_text(display_text))

        for word_data in best.get("Words", []):
            offset = word_data.get("Offset")
            duration = word_data.get("Duration")
            start = ticks_to_seconds(offset)
            end = ticks_to_seconds(offset + duration) if offset is not None and duration is not None else None
            word_text = normalize_utf8_text(word_data.get("Word", ""))

            words.append(
                {
                    "word": word_text,
                    "start": start,
                    "end": end,
                }
            )

    transcript = build_transcript_with_pauses(words, " ".join(transcript_parts).strip() or fallback_text.strip())
    return transcript, words, transcript


def build_transcript_with_pauses(words: list[dict], fallback_text: str) -> str:
    if not words:
        return normalize_utf8_text(fallback_text.strip())

    parts = []
    for index, word in enumerate(words):
        current_text = str(get_word_value(word, "word", "")).strip()
        current_end = get_word_value(word, "end")
        if current_text:
            parts.append(current_text)

        if index >= len(words) - 1:
            continue

        next_start = get_word_value(words[index + 1], "start")
        if current_end is None or next_start is None:
            continue

        if float(next_start) - float(current_end) > SILENCE_PAUSE_SECONDS:
            parts.append(PAUSE_SYMBOL)

    return replace_pause_tags(" ".join(parts).strip() or fallback_text.strip())


def transcribe_audio(
    speech_key: str,
    speech_region: str,
    audio_bytes: bytes,
    filename: str,
) -> tuple[str, list[dict], str]:
    suffix = Path(filename).suffix or ".wav"
    temp_input_path = None
    temp_wav_path = None
    audio_config = None
    recognizer = None

    try:
        temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        temp_input.write(audio_bytes)
        temp_input_path = temp_input.name
        temp_input.close()

        source_audio = AudioSegment.from_file(temp_input_path)
        clean_audio = source_audio.set_frame_rate(16000).set_channels(1)

        temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        temp_wav_path = temp_wav.name
        temp_wav.close()

        clean_audio.export(temp_wav_path, format="wav")

        speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=speech_region)
        speech_config.speech_recognition_language = "tr-TR"
        speech_config.output_format = speechsdk.OutputFormat.Detailed
        word_timestamp_property = getattr(
            speechsdk.PropertyId,
            "SpeechServiceResponse_RequestWordLevelTimestamps",
            None,
        )
        if word_timestamp_property is not None:
            speech_config.set_property(word_timestamp_property, "true")

        audio_config = speechsdk.audio.AudioConfig(filename=temp_wav_path)
        recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

        done = threading.Event()
        recognized_json_results = []
        fallback_text_parts = []
        errors = []

        def handle_recognized(evt):
            if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
                if evt.result.text:
                    fallback_text_parts.append(evt.result.text)
                raw_json = evt.result.properties.get(speechsdk.PropertyId.SpeechServiceResponse_JsonResult)
                if raw_json:
                    recognized_json_results.append(json.loads(raw_json))

        def handle_canceled(evt):
            errors.append(str(evt))
            done.set()

        def handle_stopped(_evt):
            done.set()

        recognizer.recognized.connect(handle_recognized)
        recognizer.canceled.connect(handle_canceled)
        recognizer.session_stopped.connect(handle_stopped)

        recognizer.start_continuous_recognition()
        done.wait(timeout=600)
        recognizer.stop_continuous_recognition()

        if errors and not recognized_json_results:
            raise RuntimeError(f"Azure konuşma tanıma iptal edildi: {errors[0]}")

        return parse_azure_transcription_results(recognized_json_results, " ".join(fallback_text_parts))
    finally:
        if recognizer is not None:
            del recognizer
        if audio_config is not None:
            del audio_config

        for path in (temp_input_path, temp_wav_path):
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass


def get_word_value(word, key: str, default=None):
    if isinstance(word, dict):
        return word.get(key, default)
    return getattr(word, key, default)


def calculate_wpm(words) -> tuple[float, float]:
    if not words:
        return 0.0, 0.0

    timed_words = [
        word
        for word in words
        if get_word_value(word, "start") is not None and get_word_value(word, "end") is not None
    ]
    if not timed_words:
        return 0.0, 0.0

    word_count = len(timed_words)
    duration_sec = float(get_word_value(timed_words[-1], "end")) - float(get_word_value(timed_words[0], "start"))

    if duration_sec <= 0:
        return 0.0, 0.0

    wpm = word_count / (duration_sec / 60)
    return wpm, duration_sec


def split_transcript_into_time_chunks(transcribed_text: str, words, chunk_seconds: int = 40) -> list[dict]:
    timed_words = [
        word
        for word in words
        if get_word_value(word, "start") is not None and get_word_value(word, "end") is not None
    ]

    if not timed_words:
        return [{"start": None, "end": None, "text": transcribed_text}]

    chunks = []
    current_start = float(get_word_value(timed_words[0], "start"))
    current_end = current_start + chunk_seconds
    current_words = []

    for index, word in enumerate(timed_words):
        word_start = float(get_word_value(word, "start"))
        word_end = float(get_word_value(word, "end"))
        word_text = str(get_word_value(word, "word", "")).strip()

        if word_start >= current_end and current_words:
            chunks.append(
                {
                    "start": current_start,
                    "end": current_end,
                    "text": " ".join(current_words),
                }
            )
            current_start = current_end
            while word_start >= current_start + chunk_seconds:
                current_start += chunk_seconds
            current_end = current_start + chunk_seconds
            current_words = []

        if word_text:
            current_words.append(word_text)

        next_index = index + 1
        if next_index < len(timed_words):
            next_start = float(get_word_value(timed_words[next_index], "start"))
            if next_start - word_end > SILENCE_PAUSE_SECONDS:
                current_words.append(PAUSE_SYMBOL)

    if current_words:
        chunks.append(
            {
                "start": current_start,
                "end": current_end,
                "text": " ".join(current_words),
            }
        )

    return chunks or [{"start": None, "end": None, "text": transcribed_text}]


def normalize_analysis_response(data: dict) -> dict:
    if "error_timeline" not in data:
        raw_errors = data.get("hatalar") or data.get("errors") or data.get("analiz") or []
        if isinstance(raw_errors, dict):
            raw_errors = [raw_errors]

        data["error_timeline"] = [
            {
                "time": item.get("time", "-"),
                "category": item.get("category", "Fonolojik ve Ses Birleştirme Hataları"),
                "sub_error": f"ID {item.get('hata_id') or item.get('rule_id', '-') } - {item.get('tanim') or item.get('rule_definition', '-')}",
                "rule_id": item.get("hata_id") or item.get("rule_id"),
                "rule_definition": item.get("tanim") or item.get("rule_definition", "-"),
                "rule_example": item.get("rule_example") or item.get("ornek", "-"),
                "student_reading": first_token(item.get("gerceklesen") or item.get("student_reading", "-")),
                "expected_reading": first_token(item.get("beklenen") or item.get("expected_reading", "-")),
                "description": item.get("description") or item.get("aciklama", "-"),
            }
            for item in raw_errors
            if isinstance(item, dict)
        ]

    if "uzman_raporu" not in data:
        data["uzman_raporu"] = (
            "## Akademik Beceriler\n\nHam Azure transkripti ve hata kuralları üzerinden yapılandırılmış analiz üretildi.\n\n"
            "## Bilişsel Beceriler\n\nHata örüntüleri kural eşleşmelerine göre değerlendirilmelidir.\n\n"
            "## Sosyal, Duygusal ve Davranışsal Alan\n\nOkuma hatalarının öğrenci deneyimine etkisi izlenmelidir.\n\n"
            "## Aileye Öneriler\n\nKısa, düzenli ve destekleyici okuma çalışmaları önerilir.\n\n"
            "## Kısa Vadeli Hedefler\n\nÖncelikli hedef, tabloda listelenen hata türlerini azaltmaktır."
        )

    return data


def parse_time_to_seconds(value) -> float | None:
    if value in (None, "-"):
        return None

    text = str(value).strip()
    try:
        return float(text)
    except ValueError:
        pass

    if ":" in text:
        parts = text.split(":")
        try:
            if len(parts) == 2:
                return (int(parts[0]) * 60) + float(parts[1])
            if len(parts) == 3:
                return (int(parts[0]) * 3600) + (int(parts[1]) * 60) + float(parts[2])
        except ValueError:
            return None

    return None


def format_seconds_as_timestamp(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    total_seconds = max(0, int(round(seconds)))
    minutes = total_seconds // 60
    remaining_seconds = total_seconds % 60
    return f"{minutes:02d}:{remaining_seconds:02d}"


def apply_global_time_offset(timeline: list[dict], time_offset: float | None) -> list[dict]:
    if time_offset is None:
        return timeline

    for item in timeline:
        local_seconds = parse_time_to_seconds(item.get("time"))
        if local_seconds is not None:
            item["time"] = format_seconds_as_timestamp(time_offset + local_seconds)
    return timeline


def build_unified_expert_report(results: list[dict]) -> str:
    sections = [
        "## Akademik Beceriler",
        "## Bilişsel Beceriler",
        "## Sosyal, Duygusal ve Davranışsal Alan",
        "## Aileye Öneriler",
        "## Kısa Vadeli Hedefler",
    ]
    report_parts = []
    for section in sections:
        collected_sentences = []
        seen_sentences = set()
        for result in results:
            report = result.get("uzman_raporu", "")
            if section in report:
                section_body = report.split(section, 1)[1]
                next_mark = section_body.find("\n## ")
                if next_mark != -1:
                    section_body = section_body[:next_mark]
                section_body = section_body.strip()
                if section_body:
                    normalized_body = " ".join(section_body.replace("\n", " ").split())
                    for sentence in normalized_body.split("."):
                        sentence = sentence.strip()
                        if not sentence:
                            continue
                        sentence_key = sentence.lower()
                        if sentence_key not in seen_sentences:
                            collected_sentences.append(sentence)
                            seen_sentences.add(sentence_key)

        if collected_sentences:
            paragraph_text = ". ".join(collected_sentences)
            if not paragraph_text.endswith("."):
                paragraph_text += "."
            report_parts.append(f"{section}\n\n{paragraph_text}")
        else:
            report_parts.append(f"{section}\n\nBu bölüm için parça analizlerinden ayrı bir not gelmedi.")

    return "\n\n".join(report_parts)


def merge_analysis_results(results: list[dict]) -> dict:
    timeline = []

    for result in results:
        timeline.extend(result.get("error_timeline", []))

    error_counts = {category: 0 for category in ERROR_CATEGORIES}
    for item in timeline:
        category = item.get("category")
        if category in error_counts:
            error_counts[category] += 1

    return {
        "uzman_raporu": build_unified_expert_report(results),
        "hata_kategorileri": error_counts,
        "error_timeline": timeline,
        "results": results,
    }


def first_token(value: str) -> str:
    tokens = str(value or "-").strip().split()
    return tokens[0] if tokens else "-"


def request_gpt_analysis(client: OpenAI, original_text: str, transcribed_text: str, wpm: float) -> dict:
    response = client.chat.completions.create(
        model="gpt-4o",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Orijinal Metin:\n{normalize_utf8_text(original_text)}\n\n"
                    f"Azure Ham Transkript:\n{replace_pause_tags(transcribed_text)}\n\n"
                    f"Okuma Hızı (WPM): {wpm:.1f}"
                ),
            },
        ],
    )
    return normalize_analysis_response(json.loads(response.choices[0].message.content))


def synthesize_unified_expert_report(client: OpenAI, results: list[dict]) -> str:
    section_names = [
        "Akademik Beceriler",
        "Bilişsel Beceriler",
        "Sosyal, Duygusal ve Davranışsal Alan",
        "Aileye Öneriler",
        "Kısa Vadeli Hedefler",
    ]
    reports = [
        result.get("uzman_raporu", "").strip()
        for result in results
        if str(result.get("uzman_raporu", "")).strip()
    ]

    if not reports:
        return build_unified_expert_report(results)

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Sen klinik/eğitsel değerlendirme raporu editörüsün. "
                        "Sana 40 saniyelik parçalardan gelen uzman raporları verilecek. "
                        "Aynı alt başlıklar altındaki tekrar eden yorumları harmanla, gereksiz tekrarları çıkar "
                        "ve her başlık için bütünsel, kendini tekrar etmeyen tek bir paragraf üret. "
                        "Başlıkları kesinlikle koru ve yeni başlık ekleme."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Çıktıyı yalnızca şu JSON formatında ver:\n"
                        '{"uzman_raporu": "## Akademik Beceriler\\n\\nTek paragraf...\\n\\n'
                        '## Bilişsel Beceriler\\n\\nTek paragraf...\\n\\n'
                        '## Sosyal, Duygusal ve Davranışsal Alan\\n\\nTek paragraf...\\n\\n'
                        '## Aileye Öneriler\\n\\nTek paragraf...\\n\\n'
                        '## Kısa Vadeli Hedefler\\n\\nTek paragraf..."}\n\n'
                        "Korunacak başlıklar:\n"
                        + "\n".join(f"## {name}" for name in section_names)
                        + "\n\nParça raporları:\n"
                        + "\n\n--- PARÇA RAPORU ---\n\n".join(reports)
                    ),
                },
            ],
        )
        payload = json.loads(response.choices[0].message.content)
        synthesized = normalize_utf8_text(payload.get("uzman_raporu", "")).strip()
        return synthesized or build_unified_expert_report(results)
    except Exception:
        return build_unified_expert_report(results)


def generate_analysis(
    client: OpenAI,
    original_text: str,
    transcribed_text: str,
    wpm: float,
    words,
    azure_analysis_text: str,
) -> dict:
    original_text_for_gpt = normalize_utf8_text(original_text)
    transcribed_text_for_gpt = replace_pause_tags(transcribed_text)

    chunks = split_transcript_into_time_chunks(transcribed_text_for_gpt, words, chunk_seconds=40)
    analiz_sonuclari = []

    for chunk in chunks:
        if not chunk["text"].strip():
            continue

        chunk_result = request_gpt_analysis(client, original_text_for_gpt, chunk["text"], wpm)
        time_offset = chunk.get("start")
        chunk_result["error_timeline"] = apply_global_time_offset(
            chunk_result.get("error_timeline", []),
            time_offset,
        )
        analiz_sonuclari.append(chunk_result)

    if not analiz_sonuclari:
        raise ValueError("Analiz edilecek transkript parçası bulunamadı.")

    data = merge_analysis_results(analiz_sonuclari)

    error_counts = data.get("hata_kategorileri", {})
    normalized_counts = {category: int(error_counts.get(category, 0)) for category in ERROR_CATEGORIES}
    timeline = data.get("error_timeline", [])
    uzman_raporu = synthesize_unified_expert_report(client, analiz_sonuclari)

    for item in timeline:
        item["student_reading"] = first_token(item.get("student_reading", "-"))
        item["expected_reading"] = first_token(item.get("expected_reading", "-"))

    timeline_counts = {category: 0 for category in ERROR_CATEGORIES}
    for item in timeline:
        category = item.get("category")
        if category in timeline_counts:
            timeline_counts[category] += 1

    normalized_counts = {
        category: max(normalized_counts[category], timeline_counts[category])
        for category in ERROR_CATEGORIES
    }

    if not uzman_raporu.strip():
        raise ValueError('GPT-4o yanıtında "uzman_raporu" alanı bulunamadı veya boş.')

    return {
        "uzman_raporu": uzman_raporu,
        "error_counts": normalized_counts,
        "error_timeline": timeline,
        "results": data.get("results", [data]),
    }


def ensure_dejavu_fonts() -> tuple[str, str]:
    fonts_dir = Path(__file__).resolve().parent / "fonts"
    fonts_dir.mkdir(exist_ok=True)

    font_sources = {
        "DejaVuSans.ttf": (
            "https://raw.githubusercontent.com/senotrusov/dejavu-fonts-ttf/"
            "master/ttf/DejaVuSans.ttf"
        ),
        "DejaVuSans-Bold.ttf": (
            "https://raw.githubusercontent.com/senotrusov/dejavu-fonts-ttf/"
            "master/ttf/DejaVuSans-Bold.ttf"
        ),
    }

    for filename, url in font_sources.items():
        font_path = fonts_dir / filename
        if not font_path.exists():
            urllib.request.urlretrieve(url, font_path)

    return str(fonts_dir / "DejaVuSans.ttf"), str(fonts_dir / "DejaVuSans-Bold.ttf")


class ReadingReportPDF:
    NAVY = (26, 54, 93)
    TEXT = (45, 55, 72)
    MUTED = (113, 128, 150)
    BORDER = (226, 232, 240)
    ZEBRA = (247, 250, 252)
    WHITE = (255, 255, 255)

    def __init__(self):
        super().__init__()
        regular, bold = ensure_dejavu_fonts()
        self.add_font("DejaVu", "", regular, uni=True)
        self.add_font("DejaVu", "B", bold, uni=True)
        self.set_auto_page_break(auto=True, margin=15)
        self.set_margins(14, 14, 14)
        self.set_text_color(*self.TEXT)
        self.set_draw_color(*self.BORDER)
        self.set_line_width(0.2)

    def _font_style(self, bold: bool = False) -> str:
        return "B" if bold else ""

    def _ensure_space(self, height: float) -> None:
        if self.get_y() + height > self.page_break_trigger:
            self.add_page()

    def _section_title(self, title: str) -> None:
        self._ensure_space(14)
        self.set_text_color(*self.NAVY)
        self.set_font("DejaVu", self._font_style(bold=True), size=13)
        self.multi_cell(0, 8, title)
        y = self.get_y()
        self.set_draw_color(*self.NAVY)
        self.line(self.l_margin, y, self.w - self.r_margin, y)
        self.set_draw_color(*self.BORDER)
        self.set_text_color(*self.TEXT)
        self.ln(4)

    def _markdown_text(self, text: str, size: int = 11) -> None:
        for line in text.split("\n"):
            stripped = line.strip()

            if not stripped:
                self.ln(3)
                continue

            if stripped.startswith("### "):
                self._ensure_space(10)
                self.set_text_color(*self.NAVY)
                self.set_font("DejaVu", self._font_style(bold=True), size=size)
                self.multi_cell(0, 6, stripped[4:].replace("**", ""))
                self.set_text_color(*self.TEXT)
                self.ln(2)
                continue

            if stripped.startswith("## ") or stripped.startswith("# "):
                heading = stripped.lstrip("#").strip().replace("**", "")
                self._ensure_space(12)
                self.set_text_color(*self.NAVY)
                self.set_font("DejaVu", self._font_style(bold=True), size=size + 1)
                self.multi_cell(0, 7, heading)
                y = self.get_y()
                self.set_draw_color(*self.BORDER)
                self.line(self.l_margin, y, self.w - self.r_margin, y)
                self.set_text_color(*self.TEXT)
                self.ln(2)
                continue

            if stripped.startswith(("- ", "* ")):
                self._ensure_space(8)
                self.set_font("DejaVu", size=size)
                self.multi_cell(0, 6, f"• {stripped[2:].replace('**', '')}")
                self.ln(1)
                continue

            self._ensure_space(8)
            self.set_font("DejaVu", size=size)
            self.multi_cell(0, 6, stripped.replace("**", ""))
            self.ln(1)

    def _metrics_table(self, metrics: list[tuple[str, str]]) -> None:
        col_width = (self.w - self.l_margin - self.r_margin) / 2
        box_height = 11
        for label, value in metrics:
            self._ensure_space(box_height)
            x = self.get_x()
            y = self.get_y()
            self.set_fill_color(*self.ZEBRA)
            self.set_draw_color(*self.BORDER)
            self.rect(x, y, col_width, box_height, "DF")
            self.set_xy(x + 3, y + 2)
            self.set_text_color(*self.MUTED)
            self.set_font("DejaVu", size=8)
            self.multi_cell(col_width * 0.58, 4, label, border=0)
            self.set_xy(x + col_width * 0.62, y + 2)
            self.set_text_color(*self.NAVY)
            self.set_font("DejaVu", "B", size=10)
            self.multi_cell(col_width * 0.34, 5, value, border=0, align="R")
            self.set_text_color(*self.TEXT)
            self.set_xy(x, y + box_height)

    def _wrap_text_lines(self, text: str, width: float) -> list[str]:
        usable_width = max(width, 1)
        lines = []
        for paragraph in str(text or "-").split("\n"):
            words = paragraph.split()
            if not words:
                lines.append("")
                continue

            current_line = ""
            for word in words:
                if self.get_string_width(word) > usable_width:
                    if current_line:
                        lines.append(current_line)
                        current_line = ""
                    chunk = ""
                    for character in word:
                        candidate = f"{chunk}{character}"
                        if chunk and self.get_string_width(candidate) > usable_width:
                            lines.append(chunk)
                            chunk = character
                        else:
                            chunk = candidate
                    if chunk:
                        current_line = chunk
                    continue

                candidate = f"{current_line} {word}".strip()
                if current_line and self.get_string_width(candidate) > usable_width:
                    lines.append(current_line)
                    current_line = word
                else:
                    current_line = candidate

            if current_line:
                lines.append(current_line)

        return lines

    def _draw_clipped_text(self, x: float, y: float, width: float, height: float, text: str, line_height: float = 4.3) -> None:
        max_lines = max(1, int(height / line_height))
        lines = self._wrap_text_lines(text, width)
        if len(lines) > max_lines:
            lines = lines[:max_lines]
            lines[-1] = f"{lines[-1][: max(0, len(lines[-1]) - 3)]}..."

        self.set_xy(x, y)
        for line in lines:
            self.cell(width, line_height, line, border=0)
            self.set_xy(x, self.get_y() + line_height)

    def _intro_box(self, analysis: dict) -> None:
        box_height = 82
        self._ensure_space(box_height + 6)

        page_width = self.w - self.l_margin - self.r_margin
        box_width = page_width * 0.9
        x = self.l_margin + ((page_width - box_width) / 2)
        y = self.get_y()

        self.set_fill_color(*self.ZEBRA)
        self.set_draw_color(*self.BORDER)
        self.rect(x, y, box_width, box_height, "DF")

        padding = 4
        inner_x = x + padding
        inner_width = box_width - (padding * 2)
        current_y = y + padding

        self.set_text_color(*self.NAVY)
        self.set_font("DejaVu", "B", size=9)
        künye = (
            f"Öğrenci: {analysis.get('student_name', '-') or '-'} | "
            f"Sınıf: {analysis.get('student_class', '-') or '-'} | "
            f"Tarih: {analysis.get('test_date', '-') or '-'}"
        )
        self._draw_clipped_text(inner_x, current_y, inner_width, 6, künye, line_height=5)
        current_y += 9

        self.set_text_color(*self.NAVY)
        self.set_font("DejaVu", "B", size=8)
        self._draw_clipped_text(inner_x, current_y, inner_width, 5, "Orijinal Metin", line_height=4)
        current_y += 5
        self.set_text_color(*self.TEXT)
        self.set_font("DejaVu", size=7.5)
        self._draw_clipped_text(inner_x, current_y, inner_width, 24, analysis.get("original_text", ""), line_height=4)
        current_y += 27

        self.set_text_color(*self.NAVY)
        self.set_font("DejaVu", "B", size=8)
        self._draw_clipped_text(inner_x, current_y, inner_width, 5, "Öğrenci Transkripti", line_height=4)
        current_y += 5
        self.set_text_color(*self.TEXT)
        self.set_font("DejaVu", size=7.5)
        self._draw_clipped_text(inner_x, current_y, inner_width, 24, replace_pause_tags(analysis.get("transcribed_text", "")), line_height=4)

        self.set_y(y + box_height + 6)
        self.set_text_color(*self.TEXT)

    def _timeline_table(self, timeline: list[dict]) -> None:
        usable_width = self.w - 2 * self.l_margin
        col_widths = [
            usable_width * 0.05,
            usable_width * 0.04,
            usable_width * 0.08,
            usable_width * 0.22,
            usable_width * 0.10,
            usable_width * 0.10,
            usable_width * 0.41,
        ]
        headers = [
            "Zaman",
            "ID",
            "Ana Kategori",
            "Madde Tanımı",
            "Öğrencinin Okunuşu",
            "Beklenen Doğru Okunuş",
            "Açıklama",
        ]
        line_height = 6
        cell_padding = 2.2
        row_padding = (cell_padding * 2) + 2

        def estimate_cell_height(text: str, width: float) -> float:
            usable_cell_width = max(width - (cell_padding * 2), 1)
            total_lines = 0

            for paragraph in str(text).split("\n") or [""]:
                words = paragraph.split()
                if not words:
                    total_lines += 1
                    continue

                current_line = ""
                for word in words:
                    if self.get_string_width(word) > usable_cell_width:
                        if current_line:
                            total_lines += 1
                            current_line = ""
                        total_lines += max(1, int(self.get_string_width(word) / usable_cell_width) + 1)
                        continue

                    candidate = f"{current_line} {word}".strip()
                    if current_line and self.get_string_width(candidate) > usable_cell_width:
                        total_lines += 1
                        current_line = word
                    else:
                        current_line = candidate

                if current_line:
                    total_lines += 1

            return max(8, total_lines * line_height + row_padding)

        def draw_header() -> None:
            self.set_font("DejaVu", self._font_style(bold=True), size=8)
            header_height = max(
                estimate_cell_height(header, col_widths[idx])
                for idx, header in enumerate(headers)
            )
            self._ensure_space(header_height)
            x_start = self.get_x()
            y_start = self.get_y()
            for idx, header in enumerate(headers):
                x_cell = x_start + sum(col_widths[:idx])
                self.set_fill_color(*self.NAVY)
                self.set_draw_color(*self.NAVY)
                self.rect(x_cell, y_start, col_widths[idx], header_height, "DF")
                self.set_xy(x_cell + cell_padding, y_start + cell_padding)
                self.set_text_color(*self.WHITE)
                if idx in (0, 1):
                    self.cell(col_widths[idx] - (cell_padding * 2), line_height, header, border=0, align="L")
                else:
                    self.multi_cell(col_widths[idx] - (cell_padding * 2), line_height, header, border=0, align="L")
            self.set_xy(x_start, y_start + header_height)
            self.set_text_color(*self.TEXT)
            self.set_draw_color(*self.BORDER)
            self.set_font("DejaVu", size=8)

        draw_header()
        if not timeline:
            self._ensure_space(8)
            self.set_text_color(*self.MUTED)
            self.set_font("DejaVu", size=9)
            self.multi_cell(sum(col_widths), 8, "Hata kaydı bulunamadı.", border=1)
            self.set_text_color(*self.TEXT)
            self.ln(8)
            return

        self.set_font("DejaVu", size=8)
        for row_index, item in enumerate(timeline):
            rule_definition = str(item.get("rule_definition") or item.get("sub_error", "-"))
            rule_example = str(item.get("rule_example", "-"))
            if rule_example and rule_example != "-":
                rule_definition = f"{rule_definition}\nÖrnek: {rule_example}"

            row = [
                str(item.get("time", "-")),
                str(item.get("rule_id", "-")),
                str(item.get("category", "-")),
                rule_definition,
                str(item.get("student_reading", "-")),
                str(item.get("expected_reading", "-")),
                str(item.get("description", "-")),
            ]
            row_height = max(
                estimate_cell_height(value, col_widths[idx])
                for idx, value in enumerate(row)
            )

            if self.get_y() + row_height > self.page_break_trigger:
                self.add_page()
                draw_header()

            x_start = self.get_x()
            y_start = self.get_y()
            fill_color = self.ZEBRA if row_index % 2 else self.WHITE

            for idx, value in enumerate(row):
                x_cell = x_start + sum(col_widths[:idx])
                self.set_fill_color(*fill_color)
                self.set_draw_color(*self.BORDER)
                self.rect(x_cell, y_start, col_widths[idx], row_height, "DF")
                self.set_xy(x_cell + cell_padding, y_start + cell_padding)
                self.set_text_color(*self.NAVY if idx in (0, 1) else self.TEXT)
                self.set_font("DejaVu", self._font_style(bold=idx in (0, 1)), size=8)
                if idx in (0, 1):
                    self.cell(col_widths[idx] - (cell_padding * 2), line_height, value, border=0, align="L")
                else:
                    self.multi_cell(col_widths[idx] - (cell_padding * 2), line_height, value, border=0, align="L")

            self.set_text_color(*self.TEXT)
            self.set_font("DejaVu", size=8)
            self.set_xy(x_start, y_start + row_height)


def register_reportlab_fonts() -> tuple[str, str]:
    regular, bold = ensure_dejavu_fonts()
    if "DejaVu" not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(TTFont("DejaVu", regular))
    if "DejaVu-Bold" not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(TTFont("DejaVu-Bold", bold))
    return "DejaVu", "DejaVu-Bold"


def paragraph(text: str, style: ParagraphStyle) -> Paragraph:
    safe_text = str(text or "-").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    safe_text = safe_text.replace("\n", "<br/>")
    return Paragraph(safe_text, style)


def markdown_to_flowables(markdown_text: str, styles: dict) -> list:
    flowables = []
    for raw_line in str(markdown_text or "").splitlines():
        line = raw_line.strip()
        if not line:
            flowables.append(Spacer(1, 4))
            continue
        if line.startswith("## "):
            flowables.append(Spacer(1, 6))
            flowables.append(paragraph(line[3:], styles["section"]))
            continue
        if line.startswith("# "):
            flowables.append(Spacer(1, 6))
            flowables.append(paragraph(line[2:], styles["section"]))
            continue
        flowables.append(paragraph(line.replace("**", ""), styles["body"]))
    return flowables


def build_pdf(analysis: dict) -> bytes:
    regular_font, bold_font = register_reportlab_fonts()
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=landscape(A4),
        leftMargin=12 * mm,
        rightMargin=12 * mm,
        topMargin=12 * mm,
        bottomMargin=12 * mm,
    )

    base_styles = getSampleStyleSheet()
    styles = {
        "title": ParagraphStyle(
            "Title",
            parent=base_styles["Title"],
            fontName=bold_font,
            fontSize=18,
            leading=22,
            textColor=colors.HexColor("#1A365D"),
            alignment=TA_LEFT,
        ),
        "section": ParagraphStyle(
            "Section",
            parent=base_styles["Heading2"],
            fontName=bold_font,
            fontSize=12,
            leading=15,
            textColor=colors.HexColor("#1A365D"),
            alignment=TA_LEFT,
            spaceAfter=6,
        ),
        "body": ParagraphStyle(
            "Body",
            parent=base_styles["BodyText"],
            fontName=regular_font,
            fontSize=9,
            leading=13.5,
            alignment=TA_LEFT,
            wordWrap="CJK",
        ),
        "small": ParagraphStyle(
            "Small",
            parent=base_styles["BodyText"],
            fontName=regular_font,
            fontSize=7.5,
            leading=11.25,
            alignment=TA_LEFT,
            wordWrap="CJK",
        ),
        "small_bold": ParagraphStyle(
            "SmallBold",
            parent=base_styles["BodyText"],
            fontName=bold_font,
            fontSize=7.5,
            leading=11.25,
            alignment=TA_LEFT,
            wordWrap="CJK",
        ),
    }

    story = [paragraph("Periyodik Eğitsel Değerlendirme Raporu", styles["title"]), Spacer(1, 8)]

    intro_data = [
        [
            paragraph(
                (
                    f"Öğrenci: {analysis.get('student_name', '-') or '-'} | "
                    f"Sınıf: {analysis.get('student_class', '-') or '-'} | "
                    f"Tarih: {analysis.get('test_date', '-') or '-'}"
                ),
                styles["small_bold"],
            )
        ],
        [paragraph("Orijinal Metin", styles["small_bold"])],
        [paragraph(analysis.get("original_text", "-"), styles["small"])],
        [paragraph("Öğrenci Transkripti", styles["small_bold"])],
        [paragraph(replace_pause_tags(analysis.get("transcribed_text", "-")), styles["small"])],
    ]
    intro_table = Table(intro_data, colWidths=[doc.width * 0.9], hAlign="LEFT")
    intro_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F7FAFC")),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    story.extend([intro_table, Spacer(1, 10)])

    metrics = [
        ["Okuma Hızı (WPM)", f"{analysis['wpm']:.1f}"],
        ["Kelime Sayısı", str(analysis["word_count"])],
        ["Okuma Süresi (sn)", f"{analysis['duration_sec']:.1f}"],
        ["Toplam Hata", str(sum(analysis["error_counts"].values()))],
    ]
    story.append(paragraph("Genel Metrikler", styles["section"]))
    metrics_table = Table(
        [[paragraph(label, styles["small_bold"]), paragraph(value, styles["small"])] for label, value in metrics],
        colWidths=[doc.width * 0.25, doc.width * 0.25],
        hAlign="LEFT",
    )
    metrics_table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#E2E8F0")),
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F7FAFC")),
                ("PADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    story.extend([metrics_table, Spacer(1, 10)])

    story.append(paragraph("Uzman Raporu", styles["section"]))
    story.extend(markdown_to_flowables(analysis.get("uzman_raporu", ""), styles))
    story.append(Spacer(1, 10))

    story.append(paragraph("Hata Zaman Çizelgesi", styles["section"]))
    headers = ["Zaman", "ID", "Kategori", "Madde Tanımı", "Okunuş", "Beklenen", "Açıklama"]
    table_data = [[paragraph(header, styles["small_bold"]) for header in headers]]
    for item in analysis.get("error_timeline", []):
        rule_definition = item.get("rule_definition") or item.get("sub_error", "-")
        rule_example = item.get("rule_example", "-")
        if rule_example and rule_example != "-":
            rule_definition = f"{rule_definition}\nÖrnek: {rule_example}"

        table_data.append(
            [
                paragraph(item.get("time", "-"), styles["small"]),
                paragraph(item.get("rule_id", "-"), styles["small"]),
                paragraph(item.get("category", "-"), styles["small"]),
                paragraph(rule_definition, styles["small"]),
                paragraph(item.get("student_reading", "-"), styles["small"]),
                paragraph(item.get("expected_reading", "-"), styles["small"]),
                paragraph(item.get("description", "-"), styles["small"]),
            ]
        )

    table_width = doc.width
    col_widths = [
        table_width * 0.05,
        table_width * 0.04,
        table_width * 0.08,
        table_width * 0.22,
        table_width * 0.10,
        table_width * 0.10,
        table_width * 0.41,
    ]
    timeline_table = Table(table_data, colWidths=col_widths, repeatRows=1)
    timeline_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1A365D")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), bold_font),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#E2E8F0")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7FAFC")]),
            ]
        )
    )
    story.append(timeline_table)

    doc.build(story)
    return buffer.getvalue()


def build_pdf_filename(analysis: dict) -> str:
    ogrenci_adi = analysis.get("student_name") or "ogrenci"
    sinif = analysis.get("student_class") or "sinif"
    tarih = analysis.get("test_date") or "tarih"
    raw_filename = f"{ogrenci_adi}_{sinif}_{tarih}_okuma_analiz_raporu.pdf"
    return "_".join(raw_filename.split())


def render_error_metrics(error_counts: dict) -> None:
    cols = st.columns(len(ERROR_CATEGORIES))
    for col, category in zip(cols, ERROR_CATEGORIES):
        col.metric(category, error_counts.get(category, 0))


def render_timeline_table(timeline: list[dict]) -> None:
    if timeline:
        rows = []
        for item in timeline:
            rule_definition = item.get("rule_definition") or item.get("sub_error", "-")
            rule_example = item.get("rule_example", "-")
            if rule_example and rule_example != "-":
                rule_definition = f"{rule_definition}\nÖrnek: {rule_example}"

            rows.append(
                {
                    "Zaman": item.get("time", "-"),
                    "Hata ID": item.get("rule_id", "-"),
                    "Ana Kategori": item.get("category", "-"),
                    "Madde Tanımı": rule_definition,
                    "Öğrencinin Okunuşu": item.get("student_reading", "-"),
                    "Beklenen Doğru Okunuş": item.get("expected_reading", "-"),
                    "Açıklama": item.get("description", "-"),
                }
            )

        st.dataframe(
            rows,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("Bu okumada kaydedilen hata bulunamadı.")


def render_analysis_results(analysis: dict) -> None:
    st.subheader("Transkripsiyon Özeti")
    col1, col2, col3 = st.columns(3)
    col1.metric("Okuma Hızı (WPM)", f"{analysis['wpm']:.1f}")
    col2.metric("Kelime Sayısı", analysis["word_count"])
    col3.metric("Okuma Süresi (sn)", f"{analysis['duration_sec']:.1f}")

    st.subheader("Periyodik Eğitsel Değerlendirme Raporu")
    st.markdown(analysis["uzman_raporu"])

    st.markdown("#### Hata Kategorileri")
    render_error_metrics(analysis["error_counts"])

    with st.expander("Azure Ham Transkript"):
        st.write(analysis["transcribed_text"])
        if analysis.get("azure_analysis_text"):
            st.markdown("#### Tek Metin Bloğu")
            st.text(analysis["azure_analysis_text"])

    st.subheader("Hata Zaman Çizelgesi")
    render_timeline_table(analysis["error_timeline"])

    pdf_bytes = build_pdf(analysis)
    st.download_button(
        label="PDF Raporu İndir",
        data=pdf_bytes,
        file_name=build_pdf_filename(analysis),
        mime="application/pdf",
    )


st.title("Okuma ve Ses Analiz Sistemi")
st.write("Öğrencinin okuma kaydını yükleyerek hataları ve duraklamaları analiz edin.")

student_col1, student_col2, student_col3 = st.columns(3)
student_name = student_col1.text_input("Öğrenci Adı", placeholder="Ad Soyad")
student_class = student_col2.text_input("Sınıf", placeholder="Örn: 2-A")
test_date = student_col3.date_input("Test Tarihi", value=date.today())

original_text = st.text_area(
    "Okunması Beklenen Orijinal Metin",
    height=150,
    placeholder="Öğrencinin okuması beklenen metni buraya yazın...",
)

uploaded_file = st.file_uploader(
    "Ses veya Video Dosyasını Yükleyin (MP3, WAV, M4A, MP4 vb.)",
    type=["mp3", "wav", "m4a", "mp4", "aac", "ogg", "flac", "amr"],
)

if uploaded_file is not None:
    st.audio(uploaded_file)

if st.button("Analiz Et"):
    if not original_text.strip():
        st.warning("Lütfen okunması beklenen orijinal metni girin.")
    elif uploaded_file is None:
        st.warning("Lütfen bir ses dosyası yükleyin.")
    else:
        normalized_original_text = normalize_utf8_text(original_text.strip())
        openai_client = get_openai_client()
        speech_key, speech_region = get_azure_speech_credentials()

        with st.spinner("Ses dosyası Azure standart transkripsiyon ile yazıya dökülüyor..."):
            uploaded_file.seek(0)
            audio_bytes = uploaded_file.read()
            transcript_text, words, azure_analysis_text = transcribe_audio(
                speech_key,
                speech_region,
                audio_bytes,
                uploaded_file.name,
            )

        transcribed_text = normalize_utf8_text(transcript_text)
        wpm, duration_sec = calculate_wpm(words)

        with st.spinner("Disleksi uzmanı raporu hazırlanıyor..."):
            try:
                analysis_result = generate_analysis(
                    openai_client,
                    normalized_original_text,
                    transcribed_text,
                    wpm,
                    words,
                    azure_analysis_text,
                )
            except ValueError as exc:
                st.error(str(exc))
                st.stop()

        st.session_state["analysis"] = {
            "wpm": wpm,
            "word_count": len(words),
            "duration_sec": duration_sec,
            "transcribed_text": transcribed_text,
            "original_text": normalized_original_text,
            "student_name": normalize_utf8_text(student_name.strip()),
            "student_class": normalize_utf8_text(student_class.strip()),
            "test_date": test_date.strftime("%d.%m.%Y"),
            "azure_analysis_text": azure_analysis_text,
            **analysis_result,
        }

if st.session_state.get("analysis"):
    render_analysis_results(st.session_state["analysis"])
