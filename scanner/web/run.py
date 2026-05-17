"""Entry point: python -m scanner.web.run"""

import os

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("scanner.web.app:app", host="0.0.0.0", port=port, reload=False)
