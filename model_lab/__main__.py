import uvicorn


if __name__ == "__main__":
    uvicorn.run("model_lab.api:app", host="127.0.0.1", port=8010, reload=False)

