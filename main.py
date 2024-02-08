from fastapi import FastAPI, File, UploadFile, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from insightface import model_zoo
from insightface.app import FaceAnalysis
from fastapi.responses import HTMLResponse
import uvicorn
import face_recognition
from PIL import Image
from io import BytesIO
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
import asyncio
import json
import base64
import os
import pyglet


## db연결부
from database import engine, SessionLocal
from sqlalchemy.orm import Session
import models
import random
import string
from datetime import datetime
# models에 정의한 모든 클래스, 연결한 DB엔진에 테이블로 생성
models.Base.metadata.create_all(bind=engine)
# Dependency Injection(의존성 주입을 위한 함수)
# yield : FastAPI가 함수 실행을 일시 중지하고 DB 세션을 호출자에게 반환하도록 지시
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


app = FastAPI()
faceapp = FaceAnalysis()

# 정적 파일을 호스팅하기 위한 디렉토리 설정
app.mount("/static", StaticFiles(directory="static"), name="static")

# 파일에 얼굴 데이터 저장
def save_faces():
    with open("registered_faces.json", "w") as file:
        # 변환 과정에서 발생한 오류 수정: json.dump(serializable_data, file)로 변경
        serializable_data = {k: {"embedding": v["embedding"].tolist(), "image_data": base64.b64encode(v["image_data"]).decode('utf-8')} for k, v in registered_faces.items()}
        json.dump(serializable_data, file)

# 파일에서 얼굴 데이터 로드
def load_faces():
    try:
        with open("registered_faces.json", "r") as file:
            data = json.load(file)
            # JSON에서 로드한 데이터를 다시 원래 형태로 변환
            return {k: {"embedding": np.array(v["embedding"]), "image_data": bytes(v["image_data"], encoding="utf-8")} for k, v in data.items()}
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

# 등록된 얼굴 저장소 (얼굴 특징과 사용자 ID가 매핑된 딕셔너리)
registered_faces = load_faces()

# decode_image 함수 정의
def decode_image(image_bytes):
    image_data = BytesIO(image_bytes)
    return Image.open(image_data)

class FaceModel:
    def __init__(self):
        # 얼굴 인식 모델 초기화 (face_recognition 라이브러리 사용)
        self.face_recognizer = face_recognition

    def get_embedding(self, face_image):
        # 얼굴 인식
        face_locations = self.face_recognizer.face_locations(face_image)
        if not face_locations:
            raise ValueError("No face found in the image")

        # 여러 얼굴이 인식된 경우 처리
        if len(face_locations) > 1:
            raise ValueError("Multiple faces detected. Please ensure only one face is in the view.")

        # 얼굴 임베딩 추출
        face_encodings = self.face_recognizer.face_encodings(face_image, face_locations)
        if not face_encodings:
            raise ValueError("Failed to extract face embedding")

        # 첫 번째 얼굴의 임베딩 반환 (여러 얼굴이 있을 경우 수정 필요)
        return face_encodings[0]

# InsightFace 얼굴 모델 초기화
model = FaceModel()

# # InsightFace 얼굴 모델 초기화
# # CPU 모드를 위해 ctx_id=-1
# model = faceapp.prepare(ctx_id=-1)

@app.get("/", response_class=HTMLResponse)
async def get_index():
    with open("templates/index3.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content, status_code=200)

@app.get("/regist", response_class=HTMLResponse)
async def get_regist_html(request: Request):
    with open("templates/regist.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content, status_code=200)

@app.get("/enterexit", response_class=HTMLResponse)
async def get_regist_html(request: Request):
    with open("templates/enterexit.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content, status_code=200)

# Jinja2 템플릿 설정
templates = Jinja2Templates(directory="templates")

# 얼굴 등록 엔드포인트
@app.post("/register_face", response_class=HTMLResponse)
async def register_face(request: Request, image: UploadFile = File(...), user_name: str = Form(...)):
    image_bytes = await image.read()
    image_data = await asyncio.to_thread(decode_image, image_bytes)
    
    # 이미지를 NumPy 배열로 디코딩
    nparr = np.frombuffer(image_bytes, np.uint8)
    face_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if face_image is None:
        return {"error": "이미지를 디코딩하는 데 실패했습니다."}

    try:
        # 얼굴 임베딩 추출
        face_embedding = model.get_embedding(face_image)
    except ValueError as e:
        # 여러 얼굴이 감지된 경우 처리
        sound2=pyglet.resource.media('static/sounds/bad.wav',streaming=True)
        sound2.play()
        return HTMLResponse(content="다수의 인물이 인식되었습니다. 1명의 얼굴만 인식해 주세요.")

    try:
        # 사용자 ID를 추출하거나 입력받아서 등록
        registered_faces[user_name] = {"embedding": face_embedding, "image_data": image_bytes}
        save_faces()  # 변경사항을 파일에 저장
        image_to_save = Image.fromarray(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))
        image_to_save.save(f"static/images/{user_name}.jpg")

        # 무작위 알파벳 4개 + 무작위 숫자 2개로 구성된 문자열 생성 함수
        def generate_random_id():
            # 알파벳과 숫자로 구성된 문자 집합 정의
            characters = string.ascii_letters + string.digits
            # 문자 집합에서 무작위로 문자 선택하여 문자열 생성
            random_id = ''.join(random.choice(characters) for _ in range(4))  # 알파벳 4개
            random_id += ''.join(random.choice(string.digits) for _ in range(2))  # 숫자 2개
            return random_id

        # 무작위 id 생성
        random_id = generate_random_id()

        ## DB에 등록 정보 저장
        db = SessionLocal()
        newRegist = models.Regist(id=random_id, name=user_name[:-4], file_path=f"static/images/{user_name}.jpg")
        db.add(newRegist)
        db.commit()
        db.close()

        sound1=pyglet.resource.media('static/sounds/good.wav',streaming=True)
        sound1.play()
        # 등록 성공 메시지 반환
        return HTMLResponse(content=f"{user_name[:-4]}님의 이미지가 등록되었습니다!")
    except ValueError as e:
        sound2=pyglet.resource.media('static/sounds/bad.wav',streaming=True)
        sound2.play()
        return HTMLResponse(content="에러가 발생했습니다. 다시 등록해 주세요.")

# 얼굴 인식 엔드포인트
@app.post("/recognize_face", response_class=HTMLResponse)
async def recognize_face(image: UploadFile, action: str = Form(...)):
    # 이미지를 바이너리 데이터로 읽기
    image_bytes = await image.read()
    image_data = await asyncio.to_thread(decode_image, image_bytes)
    
    # 이미지를 NumPy 배열로 디코딩
    nparr = np.frombuffer(image_bytes, np.uint8)
    unknown_face_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if unknown_face_image is None:
        return {"error": "이미지를 디코딩하는 데 실패했습니다."}

    try:
        # 얼굴 임베딩 추출
        unknown_face_embedding = model.get_embedding(unknown_face_image)
    except ValueError as e:
        # 여러 얼굴이 감지된 경우 처리
        sound2=pyglet.resource.media('static/sounds/bad.wav',streaming=True)
        sound2.play()
        return HTMLResponse(content="다수의 인물이 인식되었습니다. 본인의 얼굴만 인식해 주세요.")

    # 등록된 사용자들의 얼굴과 비교
    recognized_users = []
    for user_name, registered_data in registered_faces.items():
        # 등록된 얼굴의 임베딩 데이터 추출
        registered_face_embedding = registered_data["embedding"]

        # 파일 존재 여부 확인(얼굴파일 직접 삭제 시 json에는 정보 남아있는 문제때문에)
        image_path = f"static/images/{user_name}.jpg"
        if not os.path.exists(image_path):  # 파일이 존재하지 않으면 이 사용자는 무시
            continue

        # 등록된 얼굴과의 거리 비교 (낮을수록 유사)
        distance = np.linalg.norm(registered_face_embedding - unknown_face_embedding)

        # 임계값 (거리가 얼마 이하면 출입 허용으로 판단할지 조절 가능)
        threshold = 0.3

        # 거리가 임계값 이하인 경우 출입 허용
        if distance < threshold:
            recognized_users.append(user_name)

    if recognized_users:
        name = recognized_users[0][:-4]
        # return {"message": f"환영합니다, {', '.join(recognized_users)}님!"}
        if action == "enter":

            # 현재 날짜와 시간 가져오기
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            ## DB에 등록 정보 저장
            db = SessionLocal()
            user = db.query(models.Regist).filter(models.Regist.name == name).first()
            print(user.id)
            newRecord = models.EnterExit(id=user.id, status=action)
            db.add(newRecord)
            db.commit()
            db.close()

            sound1=pyglet.resource.media('static/sounds/tada1.mp3',streaming=True)
            sound1.play()
            return HTMLResponse(content=f"입실({current_time}) : 반갑습니다, {name}님!")
        elif action == "exit":

            # 현재 날짜와 시간 가져오기
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            ## DB에 등록 정보 저장
            db = SessionLocal()
            user = db.query(models.Regist).filter(models.Regist.name == name).first()
            print(user.id)
            newRecord = models.EnterExit(id=user.id, status=action)
            db.add(newRecord)
            db.commit()
            db.close()

            sound1=pyglet.resource.media('static/sounds/bye2.mp3',streaming=True)
            sound1.play()
            return HTMLResponse(content=f"퇴실({current_time}) : {name}님, 다음에 또 봐요!")
    else:
        # return {"message": "등록되지 않은 사용자입니다."}
        sound2=pyglet.resource.media('static/sounds/bad.wav',streaming=True)
        sound2.play()
        return HTMLResponse(content="등록되지 않은 사용자입니다. 얼굴을 다시 인식해 주세요.")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)