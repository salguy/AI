from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from domain.ai import router as airouter
tags_metadata = [
    {
        "name": "inference",
        "description": "AI 추론",
    }
]

app = FastAPI(
    openapi_tags=tags_metadata
)

origins = [
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(airouter.router, tags=["ai"])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)