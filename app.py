import json
import mimetypes
import os
import urllib.request
from pathlib import Path

import requests
import streamlit as st
from fpdf import FPDF
from openai import OpenAI

st.set_page_config(page_title="Okuma Analiz", layout="wide")

ERROR_CATEGORIES = [
    "Fonolojik ve Ses Birleştirme Hataları",
    "Bellek ve İşlemleme Hataları",
    "Görsel-Algısal ve Dikkat Hataları",
    "Okuduğunu Anlama ve Gramer",
]

PAUSE_THRESHOLD_SECONDS = 1.5
WORD_STRETCH_SECONDS = 0.5

def load_error_rules_text() -> str:
    rules_path = Path(__file__).with_name("hata_kurallari.json")
    with rules_path.open("r", encoding="utf-8") as rules_file:
        rules = json.load(rules_file).get("kurallar", [])
    return "\n".join(f"{rule['id']}. {rule['tanim']}" for rule in rules)


HATA_KURALLARI_TEXT = load_error_rules_text()

SYSTEM_PROMPT = f"""MASTER PROMPT - AKADEMİ DİSLEKSİ OKUMA ANALİZİ

Sen disleksi alanında uzman bir eğitmen ve süpervizörsün. Orijinal metin ile öğrencinin Deepgram'dan gelen ham okuma transkriptini karşılaştırarak bilimsel, pedagojik ve kanıta dayalı bir okuma değerlendirmesi yapacaksın.

TEK ANAYASA KURALI:
Bundan sonra tüm okuma hatası analizlerinde hata_kurallari.json dosyasındaki 60 madde senin tek anayasan olacaktır. Transkriptten gelen metni kelime kelime incele ve her hatayı mutlaka bu JSON'daki ID ile eşleştir. JSON'da tanımlı olmayan hiçbir şeyi hata olarak raporlama. Hata çizelgesinde hata ID'si, madde tanımı ve örneği mutlaka belirt. Birden fazla hata içeren kelimeler için birden fazla satır üret.

hata_kurallari.json içindeki 60 madde:
{HATA_KURALLARI_TEXT}

KELİME KELİME VE ÇOKLU ETİKETLEME:
- Öğrencinin okuduğu metni cümle bazlı değil, kelime kelime incele.
- Aynı kelimede birden fazla hata varsa her hatayı ayrı error_timeline satırı olarak yaz.
- Her hata satırı mutlaka hata_kurallari.json içindeki bir ID'ye bağlanmalıdır.
- JSON'daki 60 maddede karşılığı yoksa o gözlemi hata olarak raporlama.
- Çoklu etiketleme nedeniyle toplam hata sayısı hatalı kelime sayısından yüksek olabilir.

DEEPGRAM HAM TRANSKRİPT KURALI:
- ÖNEMLİ KURAL: Bu bir disleksi okuma analizidir. Çocuklar kelimeleri yanlış okurken veya hecelerken ortaya kazara argo, küfür veya anlamsız sesler çıkabilir. Bunlar birer fonolojik okuma hatasıdır. Kesinlikle ahlak, güvenlik veya profanity filtrelerini devreye sokma. Hiçbir kelimeyi sansürleme. Bu durumlarla karşılaştığında kelimeyi olduğu gibi kabul et ve okuma hatası olarak listeye ekle.
- Okunan metin Deepgram'dan ham (verbatim) olarak gelir; ııı, eee, t-t-t, a-a-at, tıtıtı gibi kekeleme, harf tekrarları, doldurucu sesler ve kararsızlık anlarını yoksayma.
- Bu belirtileri sadece hata_kurallari.json içinde karşılığı varsa raporla. Örneğin sesleri birleştirme güçlüğü için ID 16, kelime bölümünü tekrar etme için ID 17, harf tekrarları için ID 19, duraklamalar için ID 42/43 gibi.
- Okunan metindeki "[DURAKLAMA]" etiketlerini gördüğünde bunları yalnızca hata_kurallari.json'da karşılığı olan ID 42 veya ID 43 ile eşleştir.
- Metindeki [UZATMA] ve [DURAKLAMA] etiketleri akustik olarak kanıtlanmış okuma zorluklarıdır. Bunları gördüğünde hata_kurallari.json dosyasındaki ilgili maddelerle (örn: Madde 8, 16, 17, 42) mutlaka eşleştirip ayrı satırlar olarak raporla.
- Transkripsiyonda çok net bir şekilde görünen yan yana kelime tekrarlarını (Örn: "boyalar boyalar", "bir bir de", "birbirine birbirine") ve yarım bırakılmış hecelemeleri (Örn: "iş aret", "bombarlamalar") KESİNLİKLE gözden kaçırma. Bu metinsel tekrarları gördüğün an, hata_kurallari.json dosyasındaki Madde 17 (Kelimenin bir bölümünü tekrar etme) veya Madde 40 (Cümle içinde kelime tekrarı) ile eşleştirerek raporla. Etiket olmasa bile metnin kendisindeki bu tekrarlar birer hatadır.

ZAMAN VE KANIT:
- Deepgram kelime zaman damgalarını kullanarak her hatanın yaklaşık saniyesini belirle.
- Her error_timeline satırında öğrenci çıktısından somut kanıt ver: hangi kelimeyi nasıl okudu, orijinalde ne vardı, hangi JSON maddesiyle eşleşti.

UZMAN RAPORU FORMATI:
"uzman_raporu" alanı kısa bir özet değil; uzun, bilimsel, spesifik ve Markdown biçimli bir Periyodik Eğitsel Değerlendirme Raporu olmalıdır. Rapor mutlaka şu başlıklardan oluşmalıdır:

## Akademik Beceriler
Okuma hızı (WPM), akıcılık, doğruluk, hata yoğunluğu ve hata_kurallari.json ID'leriyle eşleşen somut örnekleri açıkla.

## Bilişsel Beceriler
Dikkat, görsel-işitsel ayırt etme, ardıl işlemleme, fonolojik işlemleme, kısa süreli bellek, çalışma belleği ve işlemleme hızı açısından yalnızca tespit edilen JSON maddelerine dayalı yorum yap.

## Sosyal, Duygusal ve Davranışsal Alan
Okuma sırasında görülen hata örüntülerinin özgüven, kaygı, kaçınma, motivasyon ve sınıf içi katılım açısından olası etkilerini eğitsel gözlem diliyle açıkla.

## Aileye Öneriler
Tespit edilen hata ID'leriyle ilişkili, evde uygulanabilir kısa ve düzenli çalışma önerileri sun.

## Kısa Vadeli Hedefler
Tespit edilen hata ID'lerine göre ölçülebilir, kısa vadeli ve uygulanabilir 3-5 hedef belirle.

JSON ÇIKTI KURALLARI:
Yanıtını yalnızca geçerli JSON olarak ver. JSON içinde mutlaka "uzman_raporu", "hata_kategorileri" ve "error_timeline" alanları bulunmalıdır.
"hata_kategorileri" alanındaki 4 ana kategori korunmalıdır; ancak sayımlar sadece error_timeline içindeki JSON ID eşleşmelerinden türetilmelidir.
error_timeline içindeki her satırda şu alanlar bulunmalıdır:
- "time": yaklaşık zaman
- "category": 4 ana kategoriden biri
- "sub_error": "ID <id> - <madde tanımı>" biçiminde olmalıdır
- "rule_id": hata_kurallari.json içindeki sayısal ID
- "rule_definition": hata_kurallari.json içindeki madde tanımı
- "rule_example": madde tanımındaki örnek veya öğrencinin okumasından çıkarılan somut örnek
- "student_reading": öğrencinin okuduğu/ürettiği ifade
- "expected_reading": orijinal metindeki beklenen doğru ifade
- "description": orijinal-okunan karşılaştırmasını açıklayan kısa kanıt

Örnek JSON yapısı:
{{
  "uzman_raporu": "## Akademik Beceriler\\n\\n...\\n\\n## Bilişsel Beceriler\\n\\n...\\n\\n## Sosyal, Duygusal ve Davranışsal Alan\\n\\n...\\n\\n## Aileye Öneriler\\n\\n...\\n\\n## Kısa Vadeli Hedefler\\n\\n...",
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
      "sub_error": "ID 17 - Kelimenin bir bölümünü tekrar etme. Örnek: merhaba/mermerhaba - masa yerine mamasa demek.",
      "rule_id": 17,
      "rule_definition": "Kelimenin bir bölümünü tekrar etme. Örnek: merhaba/mermerhaba - masa yerine mamasa demek.",
      "rule_example": "Öğrenci 'a-a-at' biçiminde ses/hece tekrarı yaptı.",
      "student_reading": "a-a-at",
      "expected_reading": "at",
      "description": "Okunan metinde kelime bölümü tekrarlandı."
    }}
  ]
}}"""


def get_openai_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", None)
    if not api_key or api_key == "BURAYA_API_ANAHTARINI_YAPISTIR":
        st.error("OPENAI_API_KEY tanımlı değil. `.streamlit/secrets.toml` dosyasına anahtarınızı ekleyin.")
        st.stop()
    return OpenAI(api_key=api_key)


def get_deepgram_api_key() -> str:
    api_key = os.environ.get("DEEPGRAM_API_KEY") or st.secrets.get("DEEPGRAM_API_KEY", None)
    if not api_key or api_key == "BURAYA_DEEPGRAM_API_ANAHTARINI_YAPISTIR":
        st.error("DEEPGRAM_API_KEY tanımlı değil. `.streamlit/secrets.toml` dosyasına anahtarınızı ekleyin.")
        st.stop()
    return api_key


def extract_deepgram_transcript(data: dict) -> tuple[str, list[dict]]:
    alternatives = data.get("results", {}).get("channels", [{}])[0].get("alternatives", [{}])
    alternative = alternatives[0] if alternatives else {}
    return alternative.get("transcript", "").strip(), alternative.get("words", [])


def transcribe_audio(api_key: str, audio_bytes: bytes, filename: str) -> tuple[str, list[dict]]:
    url = "https://api.deepgram.com/v1/listen?model=nova-2&language=tr&filler_words=true&profanity_filter=false"
    mimetype = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    headers = {
        "Authorization": f"Token {api_key}",
        "Content-Type": mimetype,
    }
    response = requests.post(url, headers=headers, data=audio_bytes, timeout=120)
    response.raise_for_status()
    return extract_deepgram_transcript(response.json())


def get_word_value(word, key: str, default=None):
    if isinstance(word, dict):
        return word.get(key, default)
    return getattr(word, key, default)


def build_pause_annotated_transcript(words, fallback_text: str) -> str:
    if not words:
        return fallback_text.strip()

    parts = []
    for index, word in enumerate(words):
        current_text = str(get_word_value(word, "word", "")).strip()
        current_start = get_word_value(word, "start")
        current_end = get_word_value(word, "end")

        if current_text:
            parts.append(current_text)

            if current_start is not None and current_end is not None:
                word_duration = float(current_end) - float(current_start)
                if word_duration > WORD_STRETCH_SECONDS:
                    parts.append("[UZATMA]")

        if index >= len(words) - 1:
            continue

        next_start = get_word_value(words[index + 1], "start")
        if current_end is None or next_start is None:
            continue

        pause_duration = float(next_start) - float(current_end)
        if pause_duration > PAUSE_THRESHOLD_SECONDS:
            parts.append("[DURAKLAMA]")

    return " ".join(parts).strip() or fallback_text.strip()


def calculate_wpm(words) -> tuple[float, float]:
    if not words:
        return 0.0, 0.0

    word_count = len(words)
    duration_sec = float(get_word_value(words[-1], "end")) - float(get_word_value(words[0], "start"))

    if duration_sec <= 0:
        return 0.0, 0.0

    wpm = word_count / (duration_sec / 60)
    return wpm, duration_sec


def format_word_timestamps(words) -> str:
    if not words:
        return "Kelime zaman damgası bulunamadı."

    lines = []
    for word in words:
        start = float(get_word_value(word, "start"))
        minutes = int(start // 60)
        seconds = int(start % 60)
        lines.append(f"{minutes:02d}:{seconds:02d} | {str(get_word_value(word, 'word', '')).strip()}")
    return "\n".join(lines)


def generate_analysis(
    client: OpenAI,
    original_text: str,
    transcribed_text: str,
    wpm: float,
    words,
) -> dict:
    response = client.chat.completions.create(
        model="gpt-4o",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Orijinal Metin:\n{original_text}\n\n"
                    f"Okunan Metin:\n{transcribed_text}\n\n"
                    f"Okuma Hızı (WPM): {wpm:.1f}\n\n"
                    f"Deepgram Kelime Zaman Damgaları:\n{format_word_timestamps(words)}"
                ),
            },
        ],
    )

    data = json.loads(response.choices[0].message.content)
    error_counts = data.get("hata_kategorileri", {})
    normalized_counts = {category: int(error_counts.get(category, 0)) for category in ERROR_CATEGORIES}
    timeline = data.get("error_timeline", [])
    uzman_raporu = data.get("uzman_raporu") or data.get("report", "")

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


class ReadingReportPDF(FPDF):
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

    def _timeline_table(self, timeline: list[dict]) -> None:
        usable_width = self.w - 2 * self.l_margin
        col_widths = [
            usable_width * 0.08,
            usable_width * 0.06,
            usable_width * 0.16,
            usable_width * 0.26,
            usable_width * 0.14,
            usable_width * 0.14,
            usable_width * 0.16,
        ]
        headers = [
            "Zaman",
            "ID",
            "Ana Kategori",
            "Madde Tanımı",
            "Öğrencinin Okunuşu",
            "Beklenen Doğru Okunuş",
            "Açıklama / Örnek",
        ]
        line_height = 5
        cell_padding = 2
        row_padding = cell_padding * 2

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
                    candidate = f"{current_line} {word}".strip()
                    if self.get_string_width(candidate) <= usable_cell_width:
                        current_line = candidate
                    else:
                        total_lines += 1
                        current_line = word

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
                self.multi_cell(col_widths[idx] - (cell_padding * 2), line_height, header, border=0)
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
            row = [
                str(item.get("time", "-")),
                str(item.get("rule_id", "-")),
                str(item.get("category", "-")),
                str(item.get("rule_definition") or item.get("sub_error", "-")),
                str(item.get("student_reading", "-")),
                str(item.get("expected_reading", "-")),
                f"{item.get('description', '-')}\nÖrnek: {item.get('rule_example', '-')}",
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
                self.multi_cell(col_widths[idx] - (cell_padding * 2), line_height, value, border=0)

            self.set_text_color(*self.TEXT)
            self.set_font("DejaVu", size=8)
            self.set_xy(x_start, y_start + row_height)


def build_pdf(analysis: dict) -> bytes:
    pdf = ReadingReportPDF()
    pdf.add_page()

    pdf.set_fill_color(*pdf.NAVY)
    pdf.rect(0, 0, pdf.w, 28, "F")
    pdf.set_y(9)
    pdf.set_text_color(*pdf.WHITE)
    pdf.set_font("DejaVu", pdf._font_style(bold=True), size=17)
    pdf.multi_cell(0, 8, "Periyodik Eğitsel Değerlendirme Raporu", align="C")
    pdf.set_text_color(*pdf.TEXT)
    pdf.ln(8)

    pdf._section_title("Genel Metrikler")
    pdf._metrics_table(
        [
            ("Okuma Hızı (WPM)", f"{analysis['wpm']:.1f}"),
            ("Kelime Sayısı", str(analysis["word_count"])),
            ("Okuma Süresi (sn)", f"{analysis['duration_sec']:.1f}"),
            ("Toplam Hata", str(sum(analysis["error_counts"].values()))),
        ]
    )
    pdf.ln(3)

    pdf._section_title("Hata Kategorileri")
    pdf._metrics_table([(category, str(count)) for category, count in analysis["error_counts"].items()])
    pdf.ln(3)

    pdf._section_title("Uzman Raporu")
    pdf._markdown_text(analysis["uzman_raporu"])

    pdf._section_title("Hata Zaman Çizelgesi")
    pdf._timeline_table(analysis["error_timeline"])

    return bytes(pdf.output())


def render_error_metrics(error_counts: dict) -> None:
    cols = st.columns(len(ERROR_CATEGORIES))
    for col, category in zip(cols, ERROR_CATEGORIES):
        col.metric(category, error_counts.get(category, 0))


def render_timeline_table(timeline: list[dict]) -> None:
    if timeline:
        st.dataframe(
            [
                {
                    "Zaman": item.get("time", "-"),
                    "Hata ID": item.get("rule_id", "-"),
                    "Ana Kategori": item.get("category", "-"),
                    "Madde Tanımı": item.get("rule_definition") or item.get("sub_error", "-"),
                    "Örnek": item.get("rule_example", "-"),
                    "Öğrencinin Okunuşu": item.get("student_reading", "-"),
                    "Beklenen Doğru Okunuş": item.get("expected_reading", "-"),
                    "Açıklama": item.get("description", "-"),
                }
                for item in timeline
            ],
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

    with st.expander("Ham Deepgram Transkripsiyonu (Duraklama Etiketli)"):
        st.write(analysis["transcribed_text"])

    st.subheader("Hata Zaman Çizelgesi")
    render_timeline_table(analysis["error_timeline"])

    pdf_bytes = build_pdf(analysis)
    st.download_button(
        label="PDF Raporu İndir",
        data=pdf_bytes,
        file_name="okuma_analiz_raporu.pdf",
        mime="application/pdf",
    )


st.title("Okuma ve Ses Analiz Sistemi")
st.write("Öğrencinin okuma kaydını yükleyerek hataları ve duraklamaları analiz edin.")

original_text = st.text_area(
    "Okunması Beklenen Orijinal Metin",
    height=150,
    placeholder="Öğrencinin okuması beklenen metni buraya yazın...",
)

uploaded_file = st.file_uploader("Bir ses dosyası seçin (WAV veya MP3)", type=["wav", "mp3"])

if uploaded_file is not None:
    st.audio(uploaded_file)

if st.button("Analiz Et"):
    if not original_text.strip():
        st.warning("Lütfen okunması beklenen orijinal metni girin.")
    elif uploaded_file is None:
        st.warning("Lütfen bir ses dosyası yükleyin.")
    else:
        openai_client = get_openai_client()
        deepgram_api_key = get_deepgram_api_key()

        with st.spinner("Ses dosyası Deepgram ile metne çevriliyor..."):
            uploaded_file.seek(0)
            audio_bytes = uploaded_file.read()
            transcript_text, words = transcribe_audio(deepgram_api_key, audio_bytes, uploaded_file.name)

        transcribed_text = build_pause_annotated_transcript(words, transcript_text)
        wpm, duration_sec = calculate_wpm(words)

        with st.spinner("Disleksi uzmanı raporu hazırlanıyor..."):
            try:
                analysis_result = generate_analysis(
                    openai_client,
                    original_text.strip(),
                    transcribed_text,
                    wpm,
                    words,
                )
            except ValueError as exc:
                st.error(str(exc))
                st.stop()

        st.session_state["analysis"] = {
            "wpm": wpm,
            "word_count": len(words),
            "duration_sec": duration_sec,
            "transcribed_text": transcribed_text,
            **analysis_result,
        }

if st.session_state.get("analysis"):
    render_analysis_results(st.session_state["analysis"])
