$ErrorActionPreference = "Stop"

Write-Host "=== V5.2 SETUP ===" -ForegroundColor Cyan

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    throw "Pythonia ei löytynyt. Asenna Python 3.14 ensin."
}

Write-Host "Python:"
python --version

if (-not (Test-Path ".venv")) {
    Write-Host "Luodaan .venv..."
    python -m venv .venv
}

Write-Host "Asennetaan riippuvuudet..."
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install websocket-client

if (-not (Test-Path ".env.v5")) {
    Write-Host ""
    Write-Host "Luodaan .env.v5" -ForegroundColor Yellow

    $alpacaKey = Read-Host "Alpaca API Key"
    $alpacaSecret = Read-Host "Alpaca Secret Key" -AsSecureString
    $alpacaSecretPlain = [Runtime.InteropServices.Marshal]::PtrToStringAuto(
        [Runtime.InteropServices.Marshal]::SecureStringToBSTR($alpacaSecret)
    )

    $telegramToken = Read-Host "Telegram Bot Token" -AsSecureString
    $telegramTokenPlain = [Runtime.InteropServices.Marshal]::PtrToStringAuto(
        [Runtime.InteropServices.Marshal]::SecureStringToBSTR($telegramToken)
    )

    $telegramChatId = Read-Host "Telegram Chat ID"

    @"
ALPACA_API_KEY=$alpacaKey
ALPACA_SECRET_KEY=$alpacaSecretPlain
ALPACA_PAPER=true
TELEGRAM_BOT_TOKEN=$telegramTokenPlain
TELEGRAM_CHAT_ID=$telegramChatId
"@ | Set-Content ".env.v5"

    Remove-Variable alpacaSecretPlain,telegramTokenPlain -ErrorAction SilentlyContinue

    Write-Host ".env.v5 luotu." -ForegroundColor Green
}
else {
    Write-Host ".env.v5 löytyy jo - sitä ei muuteta." -ForegroundColor Green
}

Write-Host ""
Write-Host "=== TEST ===" -ForegroundColor Cyan

.\.venv\Scripts\python.exe -c "from dotenv import load_dotenv; import os; load_dotenv('.env.v5'); print('ALPACA KEY:', bool(os.getenv('ALPACA_API_KEY'))); print('ALPACA SECRET:', bool(os.getenv('ALPACA_SECRET_KEY'))); print('PAPER:', os.getenv('ALPACA_PAPER')); print('TELEGRAM TOKEN:', bool(os.getenv('TELEGRAM_BOT_TOKEN'))); print('TELEGRAM CHAT:', bool(os.getenv('TELEGRAM_CHAT_ID')))"

Write-Host ""
Write-Host "=== SETUP VALMIS ===" -ForegroundColor Green
