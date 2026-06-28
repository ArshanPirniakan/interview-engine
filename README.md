# Interview Engine

An automated AI-powered technical interview engine built with **FastAPI** and integrated with the **Google GenAI SDK** and **OpenAI**. This backend handles session management, data validation, and real-time generation for simulated technical interviews.

---

## 🚀 Quick Start (Windows Setup)

Because Windows 11 can intercept default Python execution commands, follow these precise steps to get the server environment configured and online.

### 1. Clone & Navigate to the Project

Open PowerShell and navigate into your project root directory:

```powershell
cd C:\Users\Arsham\Desktop\interview_engine5

```

### 2. Install Dependencies

Run the package installation using the universal Windows Python launcher bypass (`py -m pip`):

```powershell
py -m pip install fastapi uvicorn[standard] pydantic google-genai openai httpx watchfiles

```

### 3. Set Your Environment Variables

The application requires a valid Gemini API Key from Google AI Studio to run its core generation logic. Pass it directly into your active PowerShell session:

```powershell
$env:GOOGLE_API_KEY="AIzaSyYourActualCorrectKeyHere"

```

> ⚠️ **Note:** Ensure your key starts with `AIzaSy`. Standard service account tokens or cloud access tokens starting with other prefixes may cause the API call to hang indefinitely.

### 4. Boot Up the Server

Launch the Uvicorn development server with live-reloading enabled:

```powershell
py -m uvicorn interview_engine5:app --reload

```

---

## 🛠️ API Architecture & Interactive Testing

Once the terminal outputs `INFO: Application startup complete.`, the application is active locally.

* **Base URL:** `http://127.0.0.1:8000`
* **Interactive OpenAPI Docs:** `http://127.0.0.1:8000/docs`

### How to Run the Models Directly:

1. Open your browser and go to `http://127.0.0.1:8000/docs`.
2. Expand any of the available `POST` or `GET` endpoints.
3. Click **"Try it out"**, fill in the requested JSON body configuration, and click **"Execute"**.
4. The backend will route the request to the underlying LLM layer and return the structured ledger directly in your browser window.

---

## 📦 Tech Stack

* **Backend Framework:** FastAPI (Asynchronous Server Gateway Interface)
* **ASGI Server:** Uvicorn (with WatchFiles for real-time code reloading)
* **AI Orchestration:** Google GenAI SDK (`google-genai`) & OpenAI SDK
* **Data Validation:** Pydantic v2
* **HTTP Client:** HTTPX

---

## 🛑 Troubleshooting Windows Errors

### "Python was not found" / Microsoft Store Trap

If running `python` tries to force-open the Microsoft Store, bypass it completely by using the short global execution command prefix **`py`** instead of `python` for all steps.

Alternatively, disable the interceptors completely via:
`Windows Start Menu -> "Manage app execution aliases" -> Toggle off python.exe and python3.exe`.

### "Fatal error in launcher: Unable to create process"

If typing `pip` throws a path serialization error, it means your environment paths are mismatched. Avoid calling the broken executable directly and force Python to use its internal module utility instead:

```powershell
py -m pip install [package_name]

```

### Infinite Loading / Request Hanging on Execute

If an API request spins indefinitely without crashing or printing terminal logs, check the following:

1. Confirm your `GOOGLE_API_KEY` is active and correctly formatted.
2. Ensure you are using modern model names (e.g., `gemini-2.5-flash` or `gemini-1.5-flash`) inside your code script rather than legacy naming patterns.
