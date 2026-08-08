from __future__ import annotations

import re
from pathlib import Path
from threading import Lock
from typing import Annotated

import cv2
import numpy as np
from deepface import DeepFace
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from starlette.requests import Request


# ============================================================
# 配置
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
FACE_DB_DIR = BASE_DIR / "face_db"
FACE_DB_DIR.mkdir(exist_ok=True)

MODEL_NAME = "ArcFace"
DETECTOR_BACKEND = "opencv"
DISTANCE_METRIC = "cosine"
MAX_UPLOAD_SIZE = 10 * 1024 * 1024
DB_LOCK = Lock()

app = FastAPI(
    title="Face Library",
    version="3.0.0",
    description="单文件：FastAPI + DeepFace/ArcFace + Bootstrap 前端",
)


# ============================================================
# 统一异常
# DeepFace 在“没有检测到人脸”等输入问题上通常抛 ValueError。
# 在这里统一转成 400，接口函数本身不再重复 try/except。
# ============================================================

@app.exception_handler(ValueError)
async def value_error_handler(_: Request, exc: ValueError):
    return JSONResponse(status_code=400, content={"detail": str(exc)})


# ============================================================
# 仅保留 3 个公共函数：编号校验、图片读取、人脸校验
# ============================================================

def validate_face_id(face_id: str) -> str:
    face_id = face_id.strip()
    if not re.fullmatch(r"[\w-]{1,64}", face_id, flags=re.UNICODE):
        raise HTTPException(
            status_code=400,
            detail="人脸编号只能包含中文、字母、数字、下划线和横线，最长 64 个字符",
        )
    return face_id


async def read_image(file: UploadFile) -> np.ndarray:
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="上传文件为空")
    if len(content) > MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=413, detail="图片不能超过 10MB")

    image = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="无法解析图片，请上传 JPG / PNG / WebP")
    return image


def require_single_face(image: np.ndarray) -> dict:
    faces = DeepFace.represent(
        img_path=image,
        model_name=MODEL_NAME,
        detector_backend=DETECTOR_BACKEND,
        enforce_detection=True,
        align=True,
    )
    if len(faces) != 1:
        raise HTTPException(
            status_code=400,
            detail=f"检测到 {len(faces)} 张人脸，请上传只包含一张人脸的图片",
        )
    return faces[0]


# ============================================================
# API
# ============================================================

@app.get("/api/health")
def health():
    return {
        "ok": True,
        "model": MODEL_NAME,
        "detector": DETECTOR_BACKEND,
        "metric": DISTANCE_METRIC,
        "face_count": sum(1 for _ in FACE_DB_DIR.glob("*.jpg")),
    }


@app.post("/api/face/compare")
async def compare_faces(
    image1: Annotated[UploadFile, File(description="第一张人脸图片")],
    image2: Annotated[UploadFile, File(description="第二张人脸图片")],
):
    img1 = await read_image(image1)
    img2 = await read_image(image2)

    result = DeepFace.verify(
        img1_path=img1,
        img2_path=img2,
        model_name=MODEL_NAME,
        detector_backend=DETECTOR_BACKEND,
        distance_metric=DISTANCE_METRIC,
        enforce_detection=True,
        align=True,
        silent=True,
    )

    return {
        "ok": True,
        "same_person": bool(result["verified"]),
        "similarity": round(float(result["confidence"]), 2),
        "distance": round(float(result["distance"]), 6),
        "threshold": round(float(result["threshold"]), 6),
        "model": result.get("model", MODEL_NAME),
        "metric": result.get("similarity_metric", DISTANCE_METRIC),
        "facial_areas": result.get("facial_areas"),
        "time": result.get("time"),
    }


@app.post("/api/face/search")
async def search_face(
    image: Annotated[UploadFile, File(description="待搜索人脸")],
    top_k: Annotated[int, Form(ge=1, le=100)] = 5,
):
    if not any(FACE_DB_DIR.glob("*.jpg")):
        return {
            "ok": True,
            "matched": False,
            "best_match": None,
            "results": [],
            "message": "人脸库为空",
        }

    query = await read_image(image)
    require_single_face(query)

    with DB_LOCK:
        frames = DeepFace.find(
            img_path=query,
            db_path=str(FACE_DB_DIR),
            model_name=MODEL_NAME,
            detector_backend=DETECTOR_BACKEND,
            distance_metric=DISTANCE_METRIC,
            enforce_detection=True,
            align=True,
            similarity_search=True,
            k=top_k,
            refresh_database=True,
            silent=True,
        )

    if not frames or frames[0].empty:
        return {"ok": True, "matched": False, "best_match": None, "results": []}

    candidates = []
    for row in frames[0].to_dict("records"):
        distance = float(row["distance"])
        threshold = float(row["threshold"])
        candidates.append(
            {
                "face_id": Path(str(row["identity"])).stem,
                "matched": distance <= threshold,
                "similarity": round(float(row["confidence"]), 2),
                "distance": round(distance, 6),
                "threshold": round(threshold, 6),
            }
        )

    candidates.sort(key=lambda item: item["distance"])
    best = candidates[0]

    return {
        "ok": True,
        "matched": best["matched"],
        "best_match": best,
        "results": candidates,
    }


@app.post("/api/face/insert")
async def insert_face(
    face_id: Annotated[str, Form(description="人脸唯一编号")],
    image: Annotated[UploadFile, File(description="注册人脸图片")],
    overwrite: Annotated[bool, Form()] = False,
):
    face_id = validate_face_id(face_id)
    target = FACE_DB_DIR / f"{face_id}.jpg"
    existed = target.exists()

    if existed and not overwrite:
        raise HTTPException(status_code=409, detail=f"人脸编号 {face_id} 已存在")

    img = await read_image(image)
    face = require_single_face(img)

    with DB_LOCK:
        if not cv2.imwrite(str(target), img):
            raise HTTPException(status_code=500, detail="保存人脸图片失败")

    return {
        "ok": True,
        "face_id": face_id,
        "overwritten": existed,
        "image": target.name,
        "facial_area": face.get("facial_area"),
        "message": "人脸保存成功",
    }


@app.delete("/api/face/delete/{face_id}")
def delete_face(face_id: str):
    face_id = validate_face_id(face_id)
    target = FACE_DB_DIR / f"{face_id}.jpg"

    if not target.exists():
        raise HTTPException(status_code=404, detail=f"人脸编号 {face_id} 不存在")

    with DB_LOCK:
        target.unlink()

    return {"ok": True, "face_id": face_id, "message": "人脸删除成功"}


@app.get("/api/face/list")
def list_faces():
    face_ids = sorted(path.stem for path in FACE_DB_DIR.glob("*.jpg"))
    return {"ok": True, "count": len(face_ids), "face_ids": face_ids}


# ============================================================
# 前端
# 所有页面 URL 返回同一份 HTML，由 pathname 决定显示哪个功能区。
# 这样新增样式、导航、错误处理时只改一处。
# ============================================================

APP_HTML = r'''<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Face Library</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.8/dist/css/bootstrap.min.css" rel="stylesheet">
  <style>
    :root {
      --orange: rgb(255, 84, 0);      /* B = 0 */
      --orange-dark: rgb(216, 50, 0); /* B = 0 */
      --amber: rgb(255, 145, 0);      /* B = 0 */
      --ink: rgb(27, 20, 16);
      --paper: rgb(255, 251, 247);
      --line: rgba(80, 42, 18, .16);
      --muted: rgba(50, 31, 20, .58);
      --header: 72px;
      --bs-primary: var(--orange);
      --bs-primary-rgb: 255, 84, 0;
      --bs-link-color: var(--orange-dark);
      --bs-link-hover-color: var(--orange);
    }

    * { box-sizing: border-box; }
    html, body { margin: 0; min-height: 100%; background: var(--paper); color: var(--ink); }
    body { font-family: Inter, system-ui, -apple-system, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif; }
    a { color: inherit; }

    .topbar {
      height: var(--header);
      display: grid;
      grid-template-columns: 230px 1fr auto;
      position: sticky;
      top: 0;
      z-index: 20;
      background: rgba(255, 251, 247, .96);
      border-bottom: 1px solid var(--line);
      backdrop-filter: blur(12px);
    }

    .brand {
      display: flex;
      align-items: center;
      gap: 10px;
      padding: 0 26px;
      text-decoration: none;
      font-weight: 950;
      border-right: 1px solid var(--line);
    }

    .brand-mark { width: 12px; height: 12px; background: var(--orange); transform: rotate(45deg); }
    .nav-main { display: flex; overflow-x: auto; }
    .nav-main a {
      min-width: 112px;
      display: grid;
      place-items: center;
      text-decoration: none;
      color: var(--muted);
      font-size: 14px;
      font-weight: 850;
      border-right: 1px solid var(--line);
    }
    .nav-main a:hover { background: rgba(255,84,0,.05); color: var(--orange-dark); }
    .nav-main a.active { background: var(--orange); color: white; }
    .meta { display: flex; align-items: center; gap: 14px; padding: 0 24px; font-size: 12px; font-weight: 850; color: var(--muted); }
    .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--amber); }

    [data-page] { display: none; }
    [data-page].active-page { display: grid; }

    .home {
      min-height: calc(100vh - var(--header));
      grid-template-columns: 1fr 1fr;
      grid-template-rows: 1fr 1fr;
    }
    .home-link {
      min-height: 300px;
      position: relative;
      display: flex;
      flex-direction: column;
      justify-content: flex-end;
      padding: clamp(30px, 4vw, 64px);
      text-decoration: none;
      border-right: 1px solid var(--line);
      border-bottom: 1px solid var(--line);
      overflow: hidden;
    }
    .home-link:nth-child(even) { border-right: 0; }
    .home-link:hover { background: var(--orange); color: white; }
    .home-no { position: absolute; top: 28px; left: clamp(30px, 4vw, 64px); color: var(--orange); font-size: 12px; font-weight: 950; letter-spacing: .16em; }
    .home-link:hover .home-no, .home-link:hover .home-sub { color: white; }
    .home-title { font-size: clamp(34px, 4.4vw, 68px); line-height: .95; letter-spacing: -.06em; font-weight: 950; }
    .home-sub { margin-top: 14px; color: var(--muted); font-weight: 700; }

    .workspace {
      min-height: calc(100vh - var(--header));
      grid-template-columns: minmax(0, 56%) minmax(360px, 44%);
    }
    .work, .result { min-height: calc(100vh - var(--header)); padding: clamp(30px, 4vw, 64px); }
    .work { display: flex; flex-direction: column; background: radial-gradient(circle at 95% 2%, rgba(255,84,0,.08), transparent 32%), var(--paper); }
    .result { background: var(--ink); color: white; overflow: auto; }

    .heading { display: flex; justify-content: space-between; align-items: flex-end; gap: 28px; padding-bottom: 26px; border-bottom: 1px solid var(--line); }
    .heading small { color: var(--orange); font-weight: 950; letter-spacing: .16em; }
    .heading h1 { margin: 7px 0 0; font-size: clamp(42px, 5vw, 74px); line-height: .94; letter-spacing: -.06em; font-weight: 950; }
    .heading p { max-width: 280px; margin: 0; color: var(--muted); font-size: 14px; line-height: 1.7; }

    .form-area { flex: 1; display: flex; flex-direction: column; padding-top: 28px; }
    .uploads { flex: 1; min-height: 310px; display: grid; grid-template-columns: 1fr 1fr; }
    .upload {
      min-height: 290px;
      position: relative;
      display: flex;
      flex-direction: column;
      justify-content: center;
      cursor: pointer;
      overflow: hidden;
      border-bottom: 1px solid var(--line);
    }
    .uploads .upload:first-child { padding-right: 28px; border-right: 1px solid var(--line); }
    .uploads .upload:last-child { padding-left: 28px; }
    .upload.single { flex: 1; min-height: 340px; }
    .upload input[type=file] { position: absolute; inset: 0; width: 100%; height: 100%; opacity: 0; cursor: pointer; z-index: 4; }
    .upload strong { font-size: 56px; line-height: 1; color: var(--orange); }
    .upload b { margin-top: 12px; font-size: 22px; }
    .upload span { margin-top: 7px; color: var(--muted); font-size: 13px; }
    .preview { position: absolute; inset: 16px; width: calc(100% - 32px); height: calc(100% - 32px); object-fit: contain; background: rgba(255,251,247,.94); display: none; z-index: 2; }

    .field-row { display: grid; grid-template-columns: 190px 1fr; align-items: center; gap: 24px; padding: 22px 0; border-bottom: 1px solid var(--line); }
    .field-row label { font-weight: 900; }
    .form-control, .form-control:focus { border: 0; border-radius: 0; border-bottom: 2px solid rgba(80,42,18,.22); background: transparent; box-shadow: none; padding-left: 0; color: var(--ink); }
    .form-control:focus { border-bottom-color: var(--orange); }
    .form-check-input:checked { background-color: var(--orange); border-color: var(--orange); }
    .form-check-input:focus { border-color: var(--orange); box-shadow: 0 0 0 .2rem rgba(255,84,0,.15); }
    .check-row { padding: 18px 0 0; color: var(--muted); }

    .action {
      min-height: 70px;
      width: 100%;
      margin-top: 28px;
      border: 0;
      border-radius: 0;
      background: var(--orange);
      color: white;
      font-size: 17px;
      font-weight: 950;
    }
    .action:hover { background: var(--orange-dark); }
    .action:disabled { opacity: .55; }

    .result-head { display: flex; justify-content: space-between; padding-bottom: 17px; border-bottom: 1px solid rgba(255,145,0,.22); color: rgba(255,255,255,.48); font-size: 11px; font-weight: 900; letter-spacing: .14em; }
    .result-head span:last-child { color: var(--amber); }
    .result-main { padding: 42px 0 30px; border-bottom: 1px solid rgba(255,145,0,.22); }
    .result-main strong { display: block; color: var(--amber); font-size: clamp(58px, 7vw, 112px); line-height: .88; letter-spacing: -.07em; overflow-wrap: anywhere; }
    .result-main b { display: block; margin-top: 18px; }
    .result-main p { margin: 8px 0 0; color: rgba(255,255,255,.48); font-size: 13px; line-height: 1.7; }
    .raw { margin: 26px 0 0; color: rgba(255,255,255,.55); font-size: 12px; white-space: pre-wrap; word-break: break-word; }

    .table-dark-custom { width: 100%; margin-top: 20px; font-size: 12px; border-collapse: collapse; }
    .table-dark-custom th, .table-dark-custom td { padding: 11px 7px; border-bottom: 1px solid rgba(255,145,0,.16); text-align: left; white-space: nowrap; }
    .table-dark-custom th { color: rgba(255,255,255,.4); }
    .match { color: var(--amber); font-weight: 900; }

    .face-list { display: flex; flex-wrap: wrap; gap: 10px 18px; padding-top: 28px; }
    .face-id { min-width: 100px; padding: 8px 0; border-bottom: 2px solid rgba(255,145,0,.3); font-weight: 850; }
    .list-tools { margin-top: auto; padding-top: 42px; display: flex; justify-content: space-between; align-items: end; }
    .list-tools strong { display: block; color: var(--orange); font-size: 52px; line-height: 1; }
    .line-button { border: 0; border-bottom: 2px solid var(--orange); background: transparent; color: var(--orange-dark); font-weight: 900; padding: 8px 0; }

    .notice { position: fixed; right: 22px; bottom: 22px; z-index: 50; max-width: min(440px, calc(100vw - 44px)); padding: 13px 16px; background: var(--orange-dark); color: white; font-weight: 750; box-shadow: 0 14px 36px rgba(70,24,0,.2); display: none; }

    @media (max-width: 980px) {
      .topbar { height: auto; grid-template-columns: 1fr auto; }
      .brand { min-height: 66px; border-right: 0; }
      .nav-main { grid-column: 1 / -1; border-top: 1px solid var(--line); }
      .nav-main a { min-height: 48px; flex: 1; min-width: 90px; }
      .meta { min-height: 66px; }
      .home { grid-template-columns: 1fr; grid-template-rows: none; }
      .home-link { min-height: 250px; border-right: 0; }
      .workspace { grid-template-columns: 1fr; }
      .work, .result { min-height: 640px; }
    }

    @media (max-width: 640px) {
      .meta { display: none; }
      .topbar { grid-template-columns: 1fr; }
      .nav-main a { min-width: 72px; font-size: 12px; }
      .heading { align-items: flex-start; flex-direction: column; }
      .heading p { max-width: none; }
      .uploads { grid-template-columns: 1fr; }
      .uploads .upload:first-child { padding-right: 0; border-right: 0; }
      .uploads .upload:last-child { padding-left: 0; }
      .field-row { grid-template-columns: 1fr; gap: 8px; }
    }
  </style>
</head>
<body>
  <header class="topbar">
    <a class="brand" href="/"><span class="brand-mark"></span>FACE LIBRARY</a>
    <nav class="nav-main">
      <a href="/" data-route="/">总览</a>
      <a href="/compare" data-route="/compare">人脸对比</a>
      <a href="/search" data-route="/search">搜索</a>
      <a href="/insert" data-route="/insert">插入</a>
      <a href="/delete" data-route="/delete">删除</a>
    </nav>
    <div class="meta"><span class="dot"></span>ARCFACE · COSINE <a href="/docs">API ↗</a></div>
  </header>

  <main class="home" data-page="/">
    <a class="home-link" href="/compare"><span class="home-no">01 / VERIFY</span><span class="home-title">人脸对比</span><span class="home-sub">两张图片 · 返回相似度与判定</span></a>
    <a class="home-link" href="/search"><span class="home-no">02 / SEARCH</span><span class="home-title">搜索人脸</span><span class="home-sub">1:N Top-K · 从人脸库找编号</span></a>
    <a class="home-link" href="/insert"><span class="home-no">03 / REGISTER</span><span class="home-title">插入人脸</span><span class="home-sub">编号 + 图片 · 支持覆盖</span></a>
    <a class="home-link" href="/delete"><span class="home-no">04 / DELETE</span><span class="home-title">删除编号</span><span class="home-sub">按人脸编号删除注册图片</span></a>
  </main>

  <main class="workspace" data-page="/compare">
    <section class="work">
      <header class="heading"><div><small>01 / VERIFY</small><h1>人脸对比</h1></div><p>上传两张人脸照片，返回同一人判定、相似度、距离和阈值。</p></header>
      <form id="compareForm" class="form-area">
        <section class="uploads">
          <label class="upload"><strong>A</strong><b>第一张人脸</b><span>点击选择图片</span><input id="compare1" type="file" accept="image/*" required><img id="comparePreview1" class="preview" alt=""></label>
          <label class="upload"><strong>B</strong><b>第二张人脸</b><span>点击选择图片</span><input id="compare2" type="file" accept="image/*" required><img id="comparePreview2" class="preview" alt=""></label>
        </section>
        <button class="action" type="submit">开始人脸对比</button>
      </form>
    </section>
    <aside class="result"><div class="result-head"><span>RESULT</span><span id="compareState">WAITING</span></div><section id="compareResult" class="result-main"><strong>--</strong><b>相似度</b><p>等待提交。</p></section><pre id="compareRaw" class="raw"></pre></aside>
  </main>

  <main class="workspace" data-page="/search">
    <section class="work">
      <header class="heading"><div><small>02 / SEARCH</small><h1>搜索人脸</h1></div><p>上传一张人脸，在注册库中返回距离最近的 Top-K 编号。</p></header>
      <form id="searchForm" class="form-area">
        <label class="upload single"><strong>Q</strong><b>查询人脸</b><span>点击上传待搜索图片</span><input id="searchImage" type="file" accept="image/*" required><img id="searchPreview" class="preview" alt=""></label>
        <section class="field-row"><label for="topK">返回数量 Top-K</label><input id="topK" class="form-control form-control-lg" type="number" min="1" max="100" value="5"></section>
        <button class="action" type="submit">搜索人脸库</button>
      </form>
    </section>
    <aside class="result"><div class="result-head"><span>SEARCH RESULT</span><span id="searchState">WAITING</span></div><section id="searchResult" class="result-main"><strong>--</strong><b>最佳匹配</b><p>等待提交。</p></section><section id="searchTable"></section><pre id="searchRaw" class="raw"></pre></aside>
  </main>

  <main class="workspace" data-page="/insert">
    <section class="work">
      <header class="heading"><div><small>03 / REGISTER</small><h1>插入人脸</h1></div><p>每个人脸编号对应一张 JPG 注册图片。</p></header>
      <form id="insertForm" class="form-area">
        <section class="field-row"><label for="insertFaceId">人脸编号</label><input id="insertFaceId" class="form-control form-control-lg" maxlength="64" placeholder="例如：10001" required></section>
        <label class="upload single"><strong>+</strong><b>注册图片</b><span>要求图片中只有一张可识别人脸</span><input id="insertImage" type="file" accept="image/*" required><img id="insertPreview" class="preview" alt=""></label>
        <label class="check-row"><input id="overwrite" class="form-check-input me-2" type="checkbox">编号存在时覆盖</label>
        <button class="action" type="submit">保存人脸</button>
      </form>
    </section>
    <aside class="result"><div class="result-head"><span>REGISTER RESULT</span><span id="insertState">WAITING</span></div><section id="insertResult" class="result-main"><strong>ID</strong><b>注册状态</b><p>等待提交。</p></section><pre id="insertRaw" class="raw"></pre></aside>
  </main>

  <main class="workspace" data-page="/delete">
    <section class="work">
      <header class="heading"><div><small>04 / DELETE</small><h1>删除编号</h1></div><p>输入人脸编号，删除对应注册图片。</p></header>
      <form id="deleteForm" class="form-area" style="flex:0 0 auto">
        <section class="field-row"><label for="deleteFaceId">人脸编号</label><input id="deleteFaceId" class="form-control form-control-lg" maxlength="64" placeholder="例如：10001" required></section>
        <button class="action" type="submit">删除该编号</button>
      </form>
      <section class="list-tools"><div><small class="text-uppercase fw-bold" style="color:var(--orange)">Face Library</small><strong id="faceCount">--</strong><span class="text-secondary">个注册编号</span></div><button id="refreshList" class="line-button" type="button">刷新编号</button></section>
    </section>
    <aside class="result"><div class="result-head"><span>FACE IDS</span><span id="deleteState">WAITING</span></div><section id="faceList" class="face-list"><span style="color:rgba(255,255,255,.4)">正在读取...</span></section><pre id="deleteRaw" class="raw"></pre></aside>
  </main>

  <div id="notice" class="notice"></div>

  <script>
    const $ = id => document.getElementById(id);
    const path = location.pathname === "/" ? "/" : location.pathname.replace(/\/$/, "");

    document.querySelectorAll("[data-page]").forEach(el => el.classList.toggle("active-page", el.dataset.page === path));
    document.querySelectorAll("[data-route]").forEach(el => el.classList.toggle("active", el.dataset.route === path));

    function preview(inputId, imageId) {
      const input = $(inputId), image = $(imageId);
      if (!input) return;
      input.addEventListener("change", () => {
        const file = input.files[0];
        image.style.display = file ? "block" : "none";
        if (file) image.src = URL.createObjectURL(file);
      });
    }

    preview("compare1", "comparePreview1");
    preview("compare2", "comparePreview2");
    preview("searchImage", "searchPreview");
    preview("insertImage", "insertPreview");

    async function request(url, options = {}) {
      const response = await fetch(url, options);
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "请求失败");
      return data;
    }

    async function run(button, busyText, task) {
      const idleText = button.textContent;
      button.disabled = true;
      button.textContent = busyText;
      try {
        await task();
      } catch (error) {
        const notice = $("notice");
        notice.textContent = error.message;
        notice.style.display = "block";
        clearTimeout(window.noticeTimer);
        window.noticeTimer = setTimeout(() => notice.style.display = "none", 3200);
      } finally {
        button.disabled = false;
        button.textContent = idleText;
      }
    }

    function raw(id, data) { $(id).textContent = JSON.stringify(data, null, 2); }

    $("compareForm")?.addEventListener("submit", event => {
      event.preventDefault();
      const button = event.currentTarget.querySelector("button");
      run(button, "正在对比…", async () => {
        const form = new FormData();
        form.append("image1", $("compare1").files[0]);
        form.append("image2", $("compare2").files[0]);
        const data = await request("/api/face/compare", {method:"POST", body:form});
        $("compareState").textContent = data.same_person ? "MATCHED" : "NOT MATCHED";
        $("compareResult").innerHTML = `<strong>${data.similarity}%</strong><b>${data.same_person ? "判定为同一人" : "判定为不同人"}</b><p>Distance ${data.distance} · Threshold ${data.threshold}</p>`;
        raw("compareRaw", data);
      });
    });

    $("searchForm")?.addEventListener("submit", event => {
      event.preventDefault();
      const button = event.currentTarget.querySelector("button");
      run(button, "正在搜索…", async () => {
        const form = new FormData();
        form.append("image", $("searchImage").files[0]);
        form.append("top_k", $("topK").value || "5");
        const data = await request("/api/face/search", {method:"POST", body:form});
        const best = data.best_match;
        $("searchState").textContent = data.matched ? "MATCHED" : "NO MATCH";
        $("searchResult").innerHTML = best
          ? `<strong>${best.face_id}</strong><b>最佳匹配 · ${best.similarity}%</b><p>${best.matched ? "达到同一人阈值" : "未达到同一人阈值"} · Distance ${best.distance}</p>`
          : `<strong>NONE</strong><b>没有结果</b><p>${data.message || "没有可返回的人脸"}</p>`;
        $("searchTable").innerHTML = data.results.length ? `<table class="table-dark-custom"><thead><tr><th>#</th><th>编号</th><th>相似度</th><th>距离</th><th>阈值</th></tr></thead><tbody>${data.results.map((item,i) => `<tr><td>${i+1}</td><td class="${item.matched ? "match" : ""}">${item.face_id}</td><td>${item.similarity}%</td><td>${item.distance}</td><td>${item.threshold}</td></tr>`).join("")}</tbody></table>` : "";
        raw("searchRaw", data);
      });
    });

    $("insertForm")?.addEventListener("submit", event => {
      event.preventDefault();
      const button = event.currentTarget.querySelector("button");
      run(button, "正在保存…", async () => {
        const form = new FormData();
        form.append("face_id", $("insertFaceId").value.trim());
        form.append("image", $("insertImage").files[0]);
        form.append("overwrite", $("overwrite").checked ? "true" : "false");
        const data = await request("/api/face/insert", {method:"POST", body:form});
        $("insertState").textContent = "SUCCESS";
        $("insertResult").innerHTML = `<strong>${data.face_id}</strong><b>${data.overwritten ? "覆盖成功" : "插入成功"}</b><p>${data.image}</p>`;
        raw("insertRaw", data);
      });
    });

    async function refreshList() {
      const data = await request("/api/face/list");
      $("faceCount").textContent = data.count;
      $("deleteState").textContent = `${data.count} IDS`;
      $("faceList").innerHTML = data.face_ids.length
        ? data.face_ids.map(id => `<span class="face-id">${id}</span>`).join("")
        : `<span style="color:rgba(255,255,255,.4)">当前人脸库为空</span>`;
      return data;
    }

    $("refreshList")?.addEventListener("click", event => run(event.currentTarget, "读取中…", refreshList));

    $("deleteForm")?.addEventListener("submit", event => {
      event.preventDefault();
      const faceId = $("deleteFaceId").value.trim();
      if (!confirm(`确定删除人脸编号“${faceId}”吗？`)) return;
      const button = event.currentTarget.querySelector("button");
      run(button, "正在删除…", async () => {
        const data = await request(`/api/face/delete/${encodeURIComponent(faceId)}`, {method:"DELETE"});
        $("deleteState").textContent = "DELETED";
        raw("deleteRaw", data);
        $("deleteFaceId").value = "";
        await refreshList();
      });
    });

    if (path === "/delete") run($("refreshList"), "读取中…", refreshList);
  </script>
</body>
</html>'''


@app.get("/", response_class=HTMLResponse)
@app.get("/compare", response_class=HTMLResponse)
@app.get("/search", response_class=HTMLResponse)
@app.get("/insert", response_class=HTMLResponse)
@app.get("/delete", response_class=HTMLResponse)
def frontend():
    return APP_HTML


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
