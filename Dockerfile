FROM python:3
WORKDIR /app
COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt
EXPOSE 8080
#EXPOSE 8501
COPY * /app/
RUN python3 -m unittest test_main.py
ENTRYPOINT ["streamlit", "run"]
#ENTRYPOINT ["pyhton"]
CMD ["main.py", "--server.port=8080"]