# Jura Book PDF Processor

## Description

The `process_book.py` script automatically processes scanned multi-page book PDFs, where each PDF page is a double-page spread. The script:
- Corrects page orientation,
- Splits spreads into single pages,
- Detects and corrects text orientation,
- Extracts the rock name (Polish proper noun), region, and group,
- Saves each processed page as a PDF named after the extracted rock name,
- Creates a `meta.json` file with metadata for all pages,
- Exports data to CSV and (optionally) Google Sheets.

## Requirements
- macOS
- Python 3.10+
- Tesseract OCR (with Polish language data)
- Google Cloud Service Account (for Google Sheets export)
- OpenAI API key (for Vision API extraction, optional)

## Installation

### 1. Install Homebrew (if not already installed):
```sh
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### 2. Install Tesseract with Polish language:
```sh
brew install tesseract
brew install tesseract-lang
```

### 3. Install Python dependencies:
```sh
pip install -r requirements.txt
```

### 4. Configure OpenAI Vision API
- Go to https://platform.openai.com/api-keys
- Click "Create new secret key"
- Copy the key (starts with `sk-...`)
- Create a `.env` file in your project directory with:
  ```env
  OPENAI_API_KEY=sk-...
  ```

### 5. (Optional) Configure Google Cloud for Google Sheets export
#### a. Create a Google Cloud project
- Go to https://console.cloud.google.com/
- Click the project dropdown at the top left, then "NEW PROJECT"
- Name your project (e.g. `TopoSheets`) and click "CREATE"
- Switch to your new project

#### b. Enable APIs
- Go to "APIs & Services" > "Library"
- Enable "Google Drive API" and "Google Sheets API"

#### c. Create a Service Account
- Go to "APIs & Services" > "Credentials"
- Click "CREATE CREDENTIALS" > "Service account"
- Name it (e.g. `topo-bot`), click "CREATE AND CONTINUE", then "DONE"

#### d. Download credentials.json
- In the Service Accounts list, click your new account
- Go to the "KEYS" tab
- Click "ADD KEY" > "Create new key" > "JSON" > "CREATE"
- Move the downloaded `credentials.json` to your project directory

#### e. Share access with the service account
- Copy the service account email from `credentials.json` (field `client_email`)
- Go to https://drive.google.com/
- Share a folder or a test Google Sheet with this email (as "Editor")

#### f. Install required Python packages
```sh
pip install gspread google-auth
```

## Usage

1. Place PDF files to process in the `input/` directory.
2. Run the script:
   ```sh
   python process_book.py [--debug]
   ```
   - The `--debug` flag saves intermediate images to `output/<timestamp>/debug/`.

## Parameters
- `--debug` — saves intermediate images (e.g. detected regions, lines) to the debug directory.

## Input
- PDF files in the `input/` directory.
- Each PDF page is a double-page spread.

## Output
- The `output/<timestamp>/` directory:
  - Subdirectories by region and group (if recognized),
  - Single-page PDFs named after the rock name,
  - `meta.json` — metadata for all pages (rock name, region, group, page number, side, file path),
  - `meta_export.csv` — data for Google Sheets import,
  - (optionally) a new Google Sheet with the data (if credentials are configured).

## Dependencies
- pdfplumber
- Pillow
- pytesseract
- reportlab
- gspread
- google-auth
- python-dotenv
- numpy
- opencv-python
- requests

## Notes
- The script requires a `.env` file with your OpenAI API key for Vision API extraction:
  ```env
  OPENAI_API_KEY=sk-...
  ```
- For Google Sheets export, you need a `credentials.json` file (Google Cloud Service Account).

## Author
- Script created for automatic processing of climbing guidebook PDFs.
