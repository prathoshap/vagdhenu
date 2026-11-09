from fastapi import FastAPI
from api.chant import router as chant_router

app = FastAPI(
    title="Vagdhenu API",
    description="API layer for chanting slokas and generating audio.",
    version="1.0.0"
)

# Register all API routes
app.include_router(chant_router, prefix="/v1")

# Health check endpoint
@app.get("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
