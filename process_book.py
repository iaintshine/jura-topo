import os
import datetime
import pdfplumber
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import pytesseract
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import io
import numpy as np
import cv2
import requests
from enum import Enum, auto
from image_debugger import ImageDebugger
from dotenv import load_dotenv
import sys
from dataclasses import dataclass
import json

# --- Configuration ---
INPUT_DIR = "input"
OUTPUT_DIR = "output"
# Ensure Tesseract is in your PATH or set the command explicitly
# For example, on macOS if installed via Homebrew:
# pytesseract.pytesseract.tesseract_cmd = r'/opt/homebrew/bin/tesseract'
# Or on Linux:
# pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'
# For Polish language
TESSERACT_LANG = 'pol'

# GLOBALNY DEBUGGER
image_debugger = None

# --- Enum and Data Classes ---

class PageSide(Enum):
    LEFT = "left"
    RIGHT = "right"

class PageHalf:
    def __init__(self, page_number: int, side: PageSide, image: Image.Image):
        self.page_number = page_number
        self.side = side
        self.image = image

    def get_page_name_suffix(self) -> str:
        return f"{self.side.value}_page"

@dataclass
class RockInfo:
    region: str
    group: str
    rock_name: str
    page: PageHalf | None

    def __init__(self, region: str = "NIEZNANE", group: str = "NIEZNANE", rock_name: str = "NIEZNANE", page: PageHalf | None = None):
        self.region = region
        self.group = group
        self.rock_name = "".join(c if c.isalnum() or c in " -" else "_" for c in rock_name).strip().replace(" ", "_")
        self.page = page


# --- Helper Functions ---


def get_timestamp_str() -> str:
    """Generates a YYYYMMDDHHMMSS timestamp string."""
    return datetime.datetime.now().strftime("%Y%m%d%H%M%S")


def ensure_dir(directory_path: str) -> None:
    """Ensures a directory exists, creating it if necessary."""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    print(f"Directory ensured: {directory_path}")



def convert_pdf_page_to_image(pdf_page, resolution: int = 300) -> Image.Image | None:
    """Converts a pdfplumber page object to a PIL Image."""
    try:
        pil_image = pdf_page.to_image(resolution=resolution).original
        print(
            f"Converted PDF page to image. Original mode: {pil_image.mode}, size: {pil_image.size}")
        # Ensure image is in RGB mode for consistent processing
        if pil_image.mode == 'P' or pil_image.mode == 'L':  # Palette or Grayscale
            pil_image = pil_image.convert('RGB')
        elif pil_image.mode == 'RGBA':  # RGBA with alpha
            pil_image = pil_image.convert('RGB')  # Discard alpha
        return pil_image
    except Exception as e:
        print(f"Error converting PDF page to image: {e}")
        return None


def correct_initial_spread_orientation(image: Image.Image) -> Image.Image:
    """
    Corrects the orientation of the scanned spread using Tesseract OSD.
    """
    print("Attempting to correct initial spread orientation...")
    try:
        osd_data = pytesseract.image_to_osd(image, lang=TESSERACT_LANG)
        angle = 0
        for line in osd_data.split('\n'):
            if 'Rotate:' in line:
                angle_str = line.split(':')[1].strip()
                if angle_str:  # Ensure not empty
                    angle = int(angle_str)
                break

        if angle != 0:
            print(f"Rotating spread by {angle} degrees based on OSD.")
            return image.rotate(-angle, expand=True)
        else:
            print(
                "Spread OSD suggests no rotation needed or failed to determine reliably.")
            return image
    except pytesseract.TesseractError as e:
        print(f"Error during spread OSD: {e}. Returning original image.")
        return image
    except Exception as e:
        print(
            f"Unexpected error during spread orientation correction: {e}. Returning original image.")
        return image


def split_image_into_halves(image: Image.Image) -> tuple[Image.Image | None, Image.Image | None]:
    """
    Splits an image vertically into two halves.
    """
    print(f"Splitting image of size {image.size} into halves.")
    width, height = image.size
    midpoint = width // 2

    if midpoint <= 0 or midpoint >= width:
        print(
            f"Warning: Midpoint calculation for split is problematic ({midpoint}). Cannot split.")
        return None, None

    left_half = image.crop((0, 0, midpoint, height))
    right_half = image.crop((midpoint, 0, width, height))
    print(
        f"Left half size: {left_half.size}, Right half size: {right_half.size}")
    return left_half, right_half


def correct_page_text_orientation(image_half: Image.Image) -> Image.Image:
    """
    Detects and corrects text orientation for a single page (half-spread) to be upright.
    Uses Tesseract OSD.
    """
    print(f"Correcting text orientation for page of size {image_half.size}...")
    try:
        grayscale_image = ImageOps.grayscale(image_half)
        osd_data = pytesseract.image_to_osd(
            grayscale_image, lang=TESSERACT_LANG)

        angle = 0
        for line in osd_data.split('\n'):
            if 'Rotate:' in line:
                angle_str = line.split(':')[1].strip()
                if angle_str:  # Ensure not empty
                    angle = int(angle_str)
                break

        print(f"OSD for page half: suggested rotation {angle} degrees.")

        if angle != 0:
            print(f"Rotating page by {angle} degrees based on OSD.")
            return image_half.rotate(-angle, expand=True)
        else:
            print("Page OSD suggests no rotation needed or failed to determine reliably.")
            return image_half

    except pytesseract.TesseractError as e:
        print(
            f"Error during page OSD for orientation: {e}. Returning original image.")
        return image_half
    except Exception as e:
        print(
            f"Unexpected error during page text orientation correction: {e}. Returning original image.")
        return image_half


def extract_rock_name_from_image_placeholder(image: Image.Image, lang: str = TESSERACT_LANG) -> str:
    """
    Placeholder for rock name extraction. Uses basic Tesseract OCR.
    This function needs to be replaced with a more sophisticated solution.
    """
    print("Attempting to extract rock name (placeholder using Tesseract)...")
    default_rock_name = f"unknown_rock_{get_timestamp_str()}"
    try:
        text = pytesseract.image_to_string(image, lang=lang)
        print(f"Raw OCR text: \n---\n{text}\n---")

        lines = [line.strip() for line in text.split('\n') if line.strip()]

        rock_name = default_rock_name
        if len(lines) >= 2:
            potential_name_line = lines[1]
            if potential_name_line.isupper() and len(potential_name_line) > 2 and sum(c.isspace() for c in potential_name_line) < len(potential_name_line)/2:  # Check if mostly uppercase and not just spaces
                rock_name = potential_name_line
            else:  # Fallback to first line if second doesn't fit criteria
                if lines[0].isupper() and len(lines[0]) > 2 and sum(c.isspace() for c in lines[0]) < len(lines[0])/2:
                    rock_name = lines[0]
                else:  # If neither, try to take a few words from the second line
                    rock_name = " ".join(potential_name_line.split()[
                                         :3]) if potential_name_line else default_rock_name
        elif lines:  # If only one line detected
            if lines[0].isupper() and len(lines[0]) > 2 and sum(c.isspace() for c in lines[0]) < len(lines[0])/2:
                rock_name = lines[0]
            else:
                rock_name = " ".join(
                    lines[0].split()[:3]) if lines[0] else default_rock_name

        # Sanitize filename
        rock_name = "".join(c if c.isalnum(
        ) or c in " -" else "_" for c in rock_name).strip().replace(" ", "_")
        if not rock_name or rock_name == "_":  # if sanitization results in empty or almost empty string
            rock_name = default_rock_name  # Use a more robust default

        print(f"Extracted rock name (placeholder): {rock_name}")
        return rock_name

    except pytesseract.TesseractError as e:
        print(f"Tesseract OCR error during rock name extraction: {e}")
        return f"ocr_error_{get_timestamp_str()}"
    except Exception as e:
        print(f"Unexpected error during rock name extraction: {e}")
        return f"extraction_error_{get_timestamp_str()}"


def save_image_as_pdf(image: Image.Image, output_pdf_path: str):
    """Saves a PIL Image object to a PDF file."""
    print(f"Saving image to PDF: {output_pdf_path}")
    try:
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        elif image.mode == 'P' or image.mode == 'L':
            image = image.convert('RGB')

        img_width, img_height = image.size
        if img_width == 0 or img_height == 0:
            print(
                f"Error: Image to save has zero dimension: {img_width}x{img_height}. Cannot save PDF.")
            return

        page_width, page_height = img_width, img_height

        c = canvas.Canvas(output_pdf_path, pagesize=(page_width, page_height))

        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        img_reader = ImageReader(img_byte_arr)

        c.drawImage(img_reader, 0, 0, width=page_width, height=page_height)
        c.showPage()
        c.save()
        print(f"Successfully saved PDF: {output_pdf_path}")
    except Exception as e:
        print(f"Error saving image as PDF to {output_pdf_path}: {e}")

# --- Hybrydowa ekstrakcja nazwy skałki ---
def preprocess_image_for_ocr(image: Image.Image) -> Image.Image:
    # Konwersja do skali szarości
    gray_image = ImageOps.grayscale(image)
    # Popraw kontrast
    enhancer = ImageEnhance.Contrast(gray_image)
    contrast_image = enhancer.enhance(2.0)
    # Wyostrzenie
    sharp_image = contrast_image.filter(ImageFilter.SHARPEN)
    return sharp_image

def extract_rock_info_with_openai(image: Image.Image) -> RockInfo | None:
    """
    Wysyła obraz do OpenAI Vision API i wyciąga region, grupę oraz nazwę skałki.
    Zakłada, że pierwszy wiersz obrazu to "region | grupa", a drugi wiersz (dużymi literami) to nazwa skałki.
    """
    import base64
    import os
    api_key = os.environ.get("OPENAI_API_KEY")
    print(f"\033[91mUsing OpenAI API key: {'***' if api_key else 'Not set'}\033[0m")
    if not api_key:
        return None
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    prompt = (
        "Na podstawie obrazu wyodrębnij trzy wartości: region, grupa oraz nazwa skałki. "
        "Pierwszy wiersz na obrazie zawiera region i grupę oddzielone pionową kreską, np. 'region | grupa'. "
        "Drugi wiersz (dużymi literami) to nazwa skałki. "
        "Zwróć wynik w formacie JSON: {\"region\":..., \"group\":..., \"rock_name\":...}. "
        "Jeśli nie możesz rozpoznać którejś wartości, wpisz 'NIEZNANE'."
    )
    payload = {
        "model": "gpt-4.1-mini",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                ]
            }
        ]
    }
    try:
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        response_data = response.json()
        print(f"\033[92mOpenAI Vision API response: {response_data}\033[0m")
        if "choices" in response_data and len(response_data["choices"]) > 0:
            content = response_data["choices"][0]["message"]["content"].strip()
            import json as _json
            try:
                data = _json.loads(content)
                region = data.get("region", "NIEZNANE")
                group = data.get("group", "NIEZNANE")
                rock_name = data.get("rock_name", "NIEZNANE")
                return RockInfo(region=region, group=group, rock_name=rock_name)
            except Exception as e:
                print(f"Could not parse JSON from OpenAI response: {e}, content: {content}")
    except Exception as e:
        print(f"OpenAI Vision API error: {e}")
    return None

def extract_top_region_above_horizontal_line(image: Image.Image) -> Image.Image:
    """
    Zwraca region obrazu od góry do pierwszej grubej poziomej linii (np. pod nazwą skałki).
    Jeśli nie znajdzie linii, zwraca górne 25% obrazu.
    """
    
    return image.crop((0, 0, image.width, int(image.height * 0.1)))
    
    import cv2
    import numpy as np
    # Konwersja do OpenCV
    img_cv = np.array(image.convert('L'))
    # Rozmycie dla redukcji szumu
    blurred = cv2.GaussianBlur(img_cv, (5, 5), 0)
    # Detekcja krawędzi
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
    # HoughLinesP do wykrycia linii poziomych
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=int(image.width*0.3), maxLineGap=20)
    # Szukamy najniższej (największe y1/y2) grubej poziomej linii blisko góry
    best_y = None
    print(f"Detected {len(lines) if lines is not None else 0} horizontal lines.")
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Sprawdź czy linia jest pozioma (mały kąt nachylenia)
            if abs(x2 - x1) > image.width * 0.5:
                # Sprawdź czy linia jest "gruba" (analiza szerokości na oryginalnym obrazie)
                # Tu uproszczenie: bierzemy pierwszą od góry
                if best_y is None or min(y1, y2) < best_y:
                    best_y = min(y1, y2)
    # Jeśli znaleziono linię, zwróć region powyżej
    if best_y is not None and best_y > 10:
        return image.crop((0, 0, image.width, best_y))
    # Jeśli nie znaleziono, fallback do 25% wysokości
    return image.crop((0, 0, image.width, int(image.height * 0.1)))

def extract_rock_name_hybrid(page_half: PageHalf, image: Image.Image, lang: str = TESSERACT_LANG) -> RockInfo | None:
    # Użyj nowej funkcji do ekstrakcji regionu
    top_region = extract_top_region_above_horizontal_line(image)
    if image_debugger:
        image_debugger.save_debug_image(top_region, f"page_{page_half.page_number}_{page_half.get_page_name_suffix()}_top_region")
    
    try:
        print("Attempting to extract rock info using OpenAI Vision API...")
        rock_info = extract_rock_info_with_openai(top_region)
        print(f"OpenAI Vision API result: {rock_info}")
        rock_info.page = page_half  # Attach the page half to the result
    
    except Exception:
        return RockInfo(rock_name=f"error_openai_{get_timestamp_str()}")
    # Wybierz najlepszą nazwę
    if rock_info and rock_info.rock_name != 'NIEZNANE':
        print(f"Wybrano nazwę skałki: {rock_info.rock_name}")
        return rock_info
    return RockInfo(rock_name=f"nierozpoznana_skalka_{get_timestamp_str()}")

# --- Main Processing Logic ---


def process_pdf_file(pdf_path: str, base_output_dir: str):
    """Processes a single PDF file."""
    print(f"\nProcessing PDF: {pdf_path}")
    meta_entries = []  # List to collect meta info for this PDF
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                print(
                    f"\n--- Processing Page {i+1} of {os.path.basename(pdf_path)} ---")

                spread_image = convert_pdf_page_to_image(page)
                if spread_image is None:
                    print(f"Skipping page {i+1} due to conversion error.")
                    continue

                print("Step 1: Correcting initial spread orientation (if needed).")
                spread_image_oriented = correct_initial_spread_orientation(
                    spread_image)

                print("Step 2: Splitting spread into two pages.")
                left_half, right_half = split_image_into_halves(
                    spread_image_oriented)

                pages_to_process = []
                if left_half and left_half.size[0] > 0 and left_half.size[1] > 0:
                    pages_to_process.append(
                        PageHalf(i+1, PageSide.LEFT, left_half))
                else:
                    print(
                        f"Left half from page {i+1} is invalid or zero-sized. Skipping.")

                if right_half and right_half.size[0] > 0 and right_half.size[1] > 0:
                    pages_to_process.append(
                        PageHalf(i+1, PageSide.RIGHT, right_half))
                else:
                    print(
                        f"Right half from page {i+1} is invalid or zero-sized. Skipping.")

                if not pages_to_process:
                    print(
                        f"No valid page halves obtained from page {i+1}. Skipping.")
                    continue

                for page_half in pages_to_process:
                    page_half_image = page_half.image
                    page_name_suffix = page_half.get_page_name_suffix()
                    print(
                        f"\n-- Processing {page_name_suffix} from spread page {i+1} --")

                    print("Step 3a: Correcting text orientation of the page half.")
                    page_half_oriented = correct_page_text_orientation(
                        page_half_image)

                    print("Step 3b: Extracting rock name.")
                    rock_info = extract_rock_name_hybrid(
                        page_half, page_half_oriented, lang=TESSERACT_LANG)

                    # Prepare output directory structure: output/timestamp/region/group/
                    region_dir = rock_info.region if rock_info and rock_info.region and rock_info.region != "NIEZNANE" else None
                    group_dir = rock_info.group if rock_info and rock_info.group and rock_info.group != "NIEZNANE" else None
                    rock_name = rock_info.rock_name if rock_info and rock_info.rock_name else f"nierozpoznana_skalka_{get_timestamp_str()}"
                    # Sanitize directory names
                    if region_dir:
                        region_dir = region_dir.replace("/", "_").replace("\\", "_")
                    if group_dir:
                        group_dir = group_dir.replace("/", "_").replace("\\", "_")
                    rock_name = rock_name.replace("/", "_").replace("\\", "_")
                    # Build output_dir
                    output_dir = base_output_dir
                    if region_dir:
                        output_dir = os.path.join(output_dir, region_dir)
                    if group_dir:
                        output_dir = os.path.join(output_dir, group_dir)
                    ensure_dir(output_dir)
                    output_filename = f"{rock_name}_{i+1}_{page_name_suffix}.pdf"
                    output_pdf_path = os.path.join(output_dir, output_filename)

                    print(
                        f"Step 3c: Saving processed page to {output_pdf_path}")
                    save_image_as_pdf(page_half_oriented, output_pdf_path)

                    # Collect meta info
                    meta_entry = {
                        "rock_name": rock_info.rock_name,
                        "region": rock_info.region,
                        "group": rock_info.group,
                        "page_number": page_half.page_number,
                        "page_side": page_half.side.value,
                        "output_path": os.path.relpath(output_pdf_path, base_output_dir)
                    }
                    meta_entries.append(meta_entry)

        # Save meta.json in the base_output_dir
        meta_json_path = os.path.join(base_output_dir, "meta.json")
        with open(meta_json_path, "w", encoding="utf-8") as f:
            json.dump(meta_entries, f, ensure_ascii=False, indent=2)
        print(f"Saved meta information to {meta_json_path}")

    except Exception as e:
        print(f"Failed to process PDF {pdf_path}: {e}")


def export_meta_to_csv(base_output_dir: str):
    """
    Exports meta.json to a CSV file for Google Sheets import.
    Columns: Name, TOPO (as HTML link), region, group
    """
    import csv
    meta_json_path = os.path.join(base_output_dir, "meta.json")
    csv_path = os.path.join(base_output_dir, "meta_export.csv")
    if not os.path.exists(meta_json_path):
        print(f"meta.json not found in {base_output_dir}, skipping CSV export.")
        return
    with open(meta_json_path, "r", encoding="utf-8") as f:
        meta_entries = json.load(f)
    with open(csv_path, "w", encoding="utf-8", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Name", "TOPO", "region", "group"])
        for entry in meta_entries:
            name = entry.get("rock_name", "")
            output_path = entry.get("output_path", "")
            if not output_path.startswith("./"):
                output_path = "./" + output_path.lstrip("./")
            topo = f'<a href="{output_path}" target="blank">TOPO</a>'
            region = entry.get("region", "")
            group = entry.get("group", "")
            writer.writerow([name, topo, region, group])
    print(f"Exported meta information to CSV: {csv_path}")


def export_meta_to_google_sheet(base_output_dir: str, sheet_name: str = "Topo Export"): 
    """
    Exports meta.json to a new Google Sheet (requires gspread & google-auth, and credentials.json in project root).
    Columns: Name, TOPO (as HYPERLINK formula), region, group
    """
    try:
        import gspread
        from google.oauth2.service_account import Credentials
        meta_json_path = os.path.join(base_output_dir, "meta.json")
        if not os.path.exists(meta_json_path):
            print(f"meta.json not found in {base_output_dir}, skipping Google Sheet export.")
            return
        creds_path = "credentials.json"
        scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
        credentials = Credentials.from_service_account_file(creds_path, scopes=scopes)
        gc = gspread.authorize(credentials)
        sh = gc.create(sheet_name)
        worksheet = sh.get_worksheet(0)
        # Prepare data
        import json
        with open(meta_json_path, "r", encoding="utf-8") as f:
            meta_entries = json.load(f)
        headers = ["Name", "TOPO", "region", "group"]
        rows = []
        for entry in meta_entries:
            name = entry.get("rock_name", "")
            output_path = entry.get("output_path", "")
            if not output_path.startswith("./"):
                output_path = "./" + output_path.lstrip("./")
            topo = f'<a href="{output_path}" target="blank">TOPO</a>'
            region = entry.get("region", "")
            group = entry.get("group", "")
            rows.append([name, topo, region, group])
        worksheet.append_row(headers)
        for row in rows:
            worksheet.append_row(row)
            
        sh.share('bodziomista@gmail.com', perm_type='user', role='writer')
        print(f"Google Sheet utworzony: {sh.url}")
    except Exception as e:
        print(f"Google Sheets export failed: {e}\n(Upewnij się, że masz gspread, google-auth i credentials.json w katalogu projektu.)")


def main():
    global image_debugger
    print("Starting PDF processing script...")

    ensure_dir(INPUT_DIR)  # Ensure input directory exists
    ensure_dir(OUTPUT_DIR)

    timestamp_str = get_timestamp_str()
    current_run_output_dir = os.path.join(OUTPUT_DIR, timestamp_str)
    ensure_dir(current_run_output_dir)

    print(f"Output will be saved in: {current_run_output_dir}")

    # Inicjalizacja debuggera
    if "--debug" in sys.argv:
        image_debugger = ImageDebugger(current_run_output_dir)
    else:
        image_debugger = None

    pdf_files_found = False
    if os.path.exists(INPUT_DIR):
        for filename in os.listdir(INPUT_DIR):
            if filename.lower().endswith(".pdf"):
                pdf_files_found = True
                pdf_path = os.path.join(INPUT_DIR, filename)
                process_pdf_file(pdf_path, current_run_output_dir)

    if not pdf_files_found:
        print(f"No PDF files found in {INPUT_DIR}.")
        # Create a dummy PDF for testing if input is empty
        try:
            from reportlab.pdfgen import canvas as rcanvas
            from reportlab.lib.pagesizes import A4
            dummy_pdf_path = os.path.join(INPUT_DIR, "dummy_for_testing.pdf")
            if not os.path.exists(dummy_pdf_path):  # Create only if it doesn't exist
                c = rcanvas.Canvas(dummy_pdf_path, pagesize=A4)
                c.drawString(
                    100, 750, "This is a dummy PDF for testing the script structure.")
                c.drawString(
                    100, 730, "It has one page. Please replace with actual scanned book PDFs.")
                c.showPage()
                c.save()
                print(f"Created a dummy PDF for testing at {dummy_pdf_path}")
                print(
                    "Please re-run the script to process this dummy PDF, or add your own PDFs.")
        except Exception as e_dummy:
            print(f"Could not create dummy PDF: {e_dummy}")

    print("\nPDF processing finished.")

    if os.path.exists(current_run_output_dir):
        export_meta_to_csv(current_run_output_dir)
        export_meta_to_google_sheet(current_run_output_dir)


if __name__ == "__main__":
    load_dotenv()
    tesseract_path_comment = """
    # --- Tesseract Configuration ---
    # On macOS, if Tesseract is installed via Homebrew, it's often at /opt/homebrew/bin/tesseract
    # or /usr/local/bin/tesseract.
    # On Linux, it's often at /usr/bin/tesseract.
    # On Windows, you need to add the Tesseract installation directory to your PATH,
    # or set the path here, e.g., pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    #
    # Example for macOS (Homebrew):
    # if os.path.exists('/opt/homebrew/bin/tesseract'):
    #     pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'
    # elif os.path.exists('/usr/local/bin/tesseract'):
    #     pytesseract.pytesseract.tesseract_cmd = '/usr/local/bin/tesseract'
    #
    # Ensure you have the Polish language data file (pol.traineddata) in your Tesseract's tessdata directory.
    # You can usually find the tessdata directory alongside the Tesseract executable.
    # For Homebrew on macOS, it might be /opt/homebrew/share/tessdata/ or /usr/local/share/tessdata/
    """
    print(tesseract_path_comment)

    try:
        tesseract_version = pytesseract.get_tesseract_version()
        print(f"Tesseract version {tesseract_version} detected.")

        # Check for Polish language data
        available_langs = pytesseract.get_languages(config='')
        if 'pol' not in available_langs:
            print("Warning: Polish language data ('pol') for Tesseract not found.")
            print(f"Available languages: {available_langs}")
            print(
                "Please install Polish language data for Tesseract (e.g., 'pol.traineddata').")
            print("On macOS with Homebrew: brew install tesseract-lang")
            print(
                "Or download 'pol.traineddata' and place it in your Tesseract 'tessdata' folder.")
        else:
            print("Polish language data ('pol') for Tesseract is available.")

    except pytesseract.TesseractNotFoundError:
        print("CRITICAL ERROR: Tesseract is not installed or not in your PATH.")
        print("Please install Tesseract OCR engine: https://github.com/tesseract-ocr/tesseract")
        print("And ensure it's added to your system's PATH or set `pytesseract.pytesseract.tesseract_cmd`.")
        exit(1)
    except Exception as e:
        print(f"Could not verify Tesseract version or language data: {e}")
        # Don't exit, but warn the user.

    main()
