# Gradio/Streamlit-FastAPI-example

# 가상 환경 만들기

```python
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch -y

```

# 라이브러리 설치

```python
pip install -r requriemnets.txt

```

# 백엔드 서버 배포(FastAPI)

```python
uvicorn backend:app --reload
```

# 프론트엔드 서버 배포(Gradio)

```python
python frontend_gradio.py
```

# 프론트엔드 서버 배포(Streamlit)

```python
python frontend_streamlit.py
```
