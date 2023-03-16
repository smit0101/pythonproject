FROM python:3
WORKDIR /app
COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt
EXPOSE 8501
COPY * /app/
RUN python3 -m unittest test_main.py
ENTRYPOINT ["streamlit", "run"]
CMD ["main.py"]