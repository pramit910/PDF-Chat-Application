FROM python:3.12-slim

RUN ["streamlit", "run", "app.py"]