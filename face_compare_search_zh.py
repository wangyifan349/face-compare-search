"""\n本项目是一个单文件 Flask 人脸工具，在同一个程序中提供独立的人脸对比页面和人脸搜索页面。程序使用 SQLite 保存人员名称、原始参考图片路径和 128 维人脸特征；搜索任务在后台线程中逐条比较数据库记录，并通过 AJAX 返回实时进度。\n\n程序使用 dlib CNN 模型检测人脸，并使用 face_recognition 的 large 模型提取特征。所有上传图片均按原始字节直接保存，不缩放、不裁剪、不转换格式、不重新编码、不压缩；录入和删除人脸库记录需要 HTTP Digest 管理员认证。\n"""

from pathlib import Path
from threading import Lock, Thread
import json
import math
import sqlite3
import time
import uuid

import face_recognition
import numpy as np
from flask import Flask, Response, jsonify, redirect, render_template_string, request, send_from_directory, url_for
from flask_httpauth import HTTPDigestAuth
from PIL import Image, ImageOps, UnidentifiedImageError

# =============================================================================
# 程序配置
# =============================================================================

application = Flask(__name__)  # 创建 Flask 应用
application.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024  # 请求上限为 64MB，不改变图片内容
application.config["SECRET_KEY"] = "face-compare-search-digest-secret-2026"  # Digest 认证签名密钥

digest_authentication = HTTPDigestAuth()  # 创建 HTTP Digest 认证器
ADMINISTRATOR_USERS = {"admin": "FaceAdmin@2026", "operator": "FaceOperator@2026"}  # 内置管理员账号，可直接修改

PROJECT_DIRECTORY = Path(__file__).resolve().parent  # 当前脚本目录
DATA_DIRECTORY = PROJECT_DIRECTORY / "face_data"
FACE_IMAGE_DIRECTORY = DATA_DIRECTORY / "face_images"
TEMPORARY_IMAGE_DIRECTORY = DATA_DIRECTORY / "temporary_images"
DATABASE_PATH = DATA_DIRECTORY / "face_library.sqlite3"

ALLOWED_IMAGE_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
FACE_DETECTION_MODEL = "cnn"  # 使用 CNN 人脸检测模型
FACE_ENCODING_MODEL = "large"  # 使用 large 人脸特征模型
FACE_LOCATION_UPSAMPLE_COUNT = 1
FACE_ENCODING_JITTER_COUNT = 1
SEARCH_JOB_RETENTION_SECONDS = 3600
SEARCH_RESULT_LIMIT = 20
SEARCH_POLL_INTERVAL_MILLISECONDS = 500

SEARCH_JOBS = {}
SEARCH_JOBS_LOCK = Lock()

# =============================================================================
# 前端页面
# =============================================================================

BASE_PAGE_TEMPLATE = """
<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{{ page_title }}</title>
<style>
:root{--surface:#fffdf9;--soft:#fff4e8;--primary:#e8873d;--primary-dark:#c86625;--primary-light:#ffe1c7;--text:#3d2c22;--muted:#8a7060;--border:#f1d5bf;--danger:#b84f3e;--success:#477b55;--shadow:0 18px 45px rgba(152,92,45,.12)}
*{box-sizing:border-box}body{min-height:100vh;margin:0;color:var(--text);background:linear-gradient(135deg,#fff8f1,#ffefe0);font-family:"Microsoft YaHei","PingFang SC",Arial,sans-serif}a{color:inherit;text-decoration:none}button,input{font:inherit}
.topbar{position:sticky;top:0;z-index:20;border-bottom:1px solid var(--border);background:rgba(255,253,249,.92);backdrop-filter:blur(12px)}.topbar-inner{display:flex;max-width:1120px;margin:auto;padding:14px 22px;align-items:center;justify-content:space-between;gap:18px}.brand{color:var(--primary-dark);font-size:20px;font-weight:800}.navigation{display:flex;gap:8px}.navigation a{padding:9px 16px;border-radius:999px;color:var(--muted);font-weight:700}.navigation a.active{color:var(--primary-dark);background:var(--primary-light)}
.page{max-width:1120px;margin:auto;padding:38px 22px 60px}.hero{margin-bottom:24px}.hero h1{margin:0 0 10px;font-size:34px}.hero p,.card-description,.notice{color:var(--muted);line-height:1.75}.hero p,.card-description{margin:0}.card-description{margin-bottom:20px}.card{padding:24px;border:1px solid rgba(232,135,61,.2);border-radius:22px;background:var(--surface);box-shadow:var(--shadow)}.card+.card{margin-top:22px}.card h2{margin:0 0 8px;font-size:21px}
.upload-panel,.form-row{display:grid;grid-template-columns:1fr 1fr;gap:16px}.upload-box{position:relative;display:flex;min-height:230px;overflow:hidden;align-items:center;justify-content:center;border:2px dashed var(--border);border-radius:18px;background:var(--soft);cursor:pointer}.upload-box input,.search-placeholder input{position:absolute;inset:0;width:100%;height:100%;opacity:0;cursor:pointer}.upload-placeholder{padding:22px;color:var(--muted);line-height:1.7;text-align:center}.upload-icon{display:block;margin-bottom:8px;font-size:34px}.preview-image{display:none;width:100%;height:230px;object-fit:contain;background:white}
.text-input,.compact-upload{width:100%;height:46px;border:1px solid var(--border);border-radius:12px;background:white}.text-input{padding:0 14px;outline:none}.compact-upload{padding:10px}.text-input:focus{border-color:var(--primary);box-shadow:0 0 0 3px rgba(232,135,61,.12)}.action-row{display:flex;margin-top:20px;align-items:center;flex-wrap:wrap;gap:12px}.primary-button,.secondary-button{padding:11px 20px;border:0;border-radius:12px;font-weight:800;cursor:pointer}.primary-button{color:white;background:var(--primary);box-shadow:0 8px 18px rgba(232,135,61,.25)}.secondary-button{color:var(--primary-dark);background:var(--primary-light)}button:disabled{opacity:.55;cursor:not-allowed}
.status-text{color:var(--muted);font-size:14px}.error-text{color:var(--danger)}.success-text{color:var(--success)}.result-panel{display:none;margin-top:22px;padding:25px;border-radius:18px;background:linear-gradient(135deg,#fff1e4,#ffe3ca);text-align:center}.similarity-label{color:var(--muted);font-size:15px}.similarity-value{margin-top:8px;color:var(--primary-dark);font-size:48px;font-weight:900}
.authentication-panel{display:flex;margin-bottom:20px;padding:15px;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:12px;border:1px solid var(--border);border-radius:15px;background:var(--soft)}.authentication-text{color:var(--muted);font-size:14px;line-height:1.7}.section-divider{height:1px;margin:24px 0;background:var(--border)}.library-header{display:flex;margin-bottom:14px;align-items:center;justify-content:space-between;gap:10px}.library-count{color:var(--muted);font-size:14px}.library-list{display:grid;max-height:430px;overflow:auto;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}.library-item{display:flex;padding:10px;align-items:center;gap:10px;border:1px solid var(--border);border-radius:15px;background:var(--soft)}.library-item img{width:52px;height:52px;border-radius:12px;background:#f4dfce;object-fit:cover}.library-item-info{min-width:0;flex:1}.library-item-name{overflow:hidden;font-weight:800;text-overflow:ellipsis;white-space:nowrap}.library-item-time{margin-top:4px;color:var(--muted);font-size:12px}.icon-button{width:32px;height:32px;border:0;border-radius:10px;color:var(--danger);background:#f8ddd3;font-weight:900;cursor:pointer}.empty-state{padding:28px;border:1px dashed var(--border);border-radius:15px;color:var(--muted);text-align:center;grid-column:1/-1}
.search-layout{display:grid;grid-template-columns:360px 1fr;gap:22px}.search-preview{display:none;width:100%;height:300px;border:1px solid var(--border);border-radius:18px;background:white;object-fit:contain;cursor:pointer}.search-placeholder{position:relative;display:flex;height:300px;padding:24px;align-items:center;justify-content:center;border:2px dashed var(--border);border-radius:18px;background:var(--soft);color:var(--muted);line-height:1.8;text-align:center;cursor:pointer}.progress-area{display:none;margin-top:20px}.progress-track{height:14px;overflow:hidden;border-radius:999px;background:#f3dfcf}.progress-bar{width:0;height:100%;background:linear-gradient(90deg,var(--primary),#f5aa69);transition:width .25s}.progress-meta{display:flex;margin-top:9px;justify-content:space-between;color:var(--muted);font-size:14px}.current-person{min-height:22px;margin-top:9px;color:var(--primary-dark);font-weight:700}.result-list{display:grid;margin-top:18px;gap:10px}.result-item{display:grid;padding:10px 14px;align-items:center;grid-template-columns:56px 1fr auto;gap:12px;border:1px solid var(--border);border-radius:15px;background:var(--soft)}.result-item img{width:56px;height:56px;border-radius:12px;background:#f3dfcf;object-fit:cover}.result-name{font-weight:800}.result-rank{margin-top:4px;color:var(--muted);font-size:12px}.result-similarity{color:var(--primary-dark);font-size:22px;font-weight:900}.notice{margin-top:14px;font-size:13px}
@media(max-width:850px){.search-layout{grid-template-columns:1fr}.library-list{grid-template-columns:repeat(2,minmax(0,1fr))}}@media(max-width:560px){.page{padding:24px 14px 45px}.brand{font-size:17px}.navigation a{padding:8px 11px}.hero h1{font-size:28px}.upload-panel,.form-row,.library-list{grid-template-columns:1fr}.similarity-value{font-size:40px}}
</style>
</head>
<body>
<header class="topbar"><div class="topbar-inner"><a class="brand" href="/compare">Face Reco 在线工具</a><nav class="navigation">{{ navigation_html|safe }}</nav></div></header>
<main class="page">{{ page_content|safe }}</main>
<script>{{ page_script|safe }}</script>
</body>
</html>
"""

COMPARE_PAGE_CONTENT = """
<section class="hero"><h1>人脸对比</h1><p>上传两张只包含一张清晰人脸的原始照片，系统使用 CNN 检测和 large 模型计算相似度。</p></section>
<section class="card">
<div class="upload-panel">
<label class="upload-box"><input id="firstFaceImage" type="file" accept="image/jpeg,image/png,image/webp"><div id="firstPlaceholder" class="upload-placeholder"><span class="upload-icon">①</span>上传第一张原始照片</div><img id="firstPreview" class="preview-image" alt="上传第一张原始照片"></label>
<label class="upload-box"><input id="secondFaceImage" type="file" accept="image/jpeg,image/png,image/webp"><div id="secondPlaceholder" class="upload-placeholder"><span class="upload-icon">②</span>上传第二张原始照片</div><img id="secondPreview" class="preview-image" alt="上传第二张原始照片"></label>
</div>
<div class="action-row"><button id="compareButton" class="primary-button" type="button">开始对比</button><span id="compareStatus" class="status-text"></span></div>
<div id="compareResult" class="result-panel"><div class="similarity-label">人脸相似度</div><div id="compareSimilarity" class="similarity-value">0.00%</div></div>
<p class="notice">服务器按原始字节保存上传文件，不缩放、不重编码、不压缩；识别时只在内存中修正方向并转换为 RGB。相似度是展示分数，不是身份认证概率。</p>
</section>
"""

SEARCH_PAGE_CONTENT = """
<section class="hero"><h1>人脸搜索</h1><p>先录入参考人脸，再上传查询照片。后台会按数据库顺序逐条计算相似度，页面使用 AJAX 显示进度和结果。</p></section>
<section class="card">
<h2>人脸库管理</h2><p class="card-description">每条记录保存姓名、原始参考图片和预先提取的 128 维特征；同一人员可录入多张参考照片。</p>
<div class="authentication-panel"><div class="authentication-text">录入和删除人脸需要 HTTP Digest 管理员认证。</div><button id="authenticateButton" class="secondary-button" type="button">管理员认证</button></div>
<div class="form-row"><input id="personName" class="text-input" type="text" maxlength="80" placeholder="输入人员姓名"><input id="libraryFaceImage" class="compact-upload" type="file" accept="image/jpeg,image/png,image/webp"></div>
<div class="action-row"><button id="addFaceButton" class="primary-button" type="button">录入人脸</button><span id="libraryStatus" class="status-text"></span></div>
<div class="section-divider"></div><div class="library-header"><h2>已录入人员</h2><span id="libraryCount" class="library-count">0 条参考照片</span></div>
<div id="libraryList" class="library-list"><div class="empty-state">正在读取人脸库…</div></div>
</section>
<section class="card">
<h2>逐个人脸搜索</h2><p class="card-description">搜索阶段按顺序比较已经保存的人脸特征，不会重新检测人脸库中的图片。</p>
<div class="search-layout"><div><label id="searchPlaceholder" class="search-placeholder">点击选择一张原始查询照片<br>照片中必须只有一张清晰人脸<input id="searchFaceImage" type="file" accept="image/jpeg,image/png,image/webp"></label><img id="searchPreview" class="search-preview" alt="查询照片预览"><div class="action-row"><button id="startSearchButton" class="primary-button" type="button">开始逐个搜索</button></div></div>
<div><div id="searchMessage" class="status-text">等待开始搜索</div><div id="progressArea" class="progress-area"><div class="progress-track"><div id="progressBar" class="progress-bar"></div></div><div class="progress-meta"><span id="progressText">0 / 0</span><span id="progressPercentage">0%</span></div><div id="currentPerson" class="current-person"></div></div><div id="searchResults" class="result-list"></div></div></div>
<p class="notice">结果仅按相似度从高到低排列，不输出匹配结论。正式身份核验应增加活体检测和人工复核。</p>
</section>
"""

ADMIN_AUTHENTICATION_PAGE = """
<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>管理员认证成功</title>
<style>body{display:flex;min-height:100vh;margin:0;align-items:center;justify-content:center;color:#3d2c22;background:#fff4e8;font-family:Arial,sans-serif}.card{width:min(440px,calc(100% - 32px));padding:30px;border:1px solid #f1d5bf;border-radius:20px;background:#fffdf9;box-shadow:0 18px 45px rgba(152,92,45,.12);text-align:center}h1{margin-top:0;color:#c86625}p{color:#8a7060;line-height:1.8}button{padding:10px 20px;border:0;border-radius:12px;color:white;background:#e8873d;font:inherit;font-weight:700;cursor:pointer}</style></head>
<body><div class="card"><h1>管理员认证成功</h1><p>当前管理员：{{ administrator_username }}。现在可以返回搜索页面录入或删除人脸。</p><button type="button" onclick="window.close()">关闭窗口</button></div></body></html>
"""

COMPARE_PAGE_SCRIPT = """
const firstFaceInput=document.getElementById("firstFaceImage"),secondFaceInput=document.getElementById("secondFaceImage"),compareButton=document.getElementById("compareButton");
function bindPreview(fileInput,previewImage,placeholder){fileInput.addEventListener("change",function(){const selectedFile=fileInput.files[0];if(!selectedFile){previewImage.style.display="none";placeholder.style.display="block";return;}previewImage.src=URL.createObjectURL(selectedFile);previewImage.style.display="block";placeholder.style.display="none";});}
async function readJsonResponse(response){let responseData={};try{responseData=await response.json();}catch(error){responseData={};}if(response.ok){return responseData;}throw new Error(responseData.message||"请求处理失败");}
async function compareSelectedFaces(){const firstFaceFile=firstFaceInput.files[0],secondFaceFile=secondFaceInput.files[0],statusElement=document.getElementById("compareStatus"),resultElement=document.getElementById("compareResult");if(!firstFaceFile||!secondFaceFile){statusElement.textContent="请先选择两张照片";statusElement.className="status-text error-text";return;}compareButton.disabled=true;resultElement.style.display="none";statusElement.textContent="正在使用 CNN 检测并提取 large 人脸特征…";statusElement.className="status-text";const requestData=new FormData();requestData.append("first_face_image",firstFaceFile);requestData.append("second_face_image",secondFaceFile);try{const response=await fetch("/api/compare",{method:"POST",body:requestData});const responseData=await readJsonResponse(response);document.getElementById("compareSimilarity").textContent=Number(responseData.similarity).toFixed(2)+"%";resultElement.style.display="block";statusElement.textContent="对比完成";statusElement.className="status-text success-text";}catch(error){statusElement.textContent=error.message;statusElement.className="status-text error-text";}finally{compareButton.disabled=false;}}
bindPreview(firstFaceInput,document.getElementById("firstPreview"),document.getElementById("firstPlaceholder"));bindPreview(secondFaceInput,document.getElementById("secondPreview"),document.getElementById("secondPlaceholder"));compareButton.addEventListener("click",compareSelectedFaces);
"""

SEARCH_PAGE_SCRIPT = """
const authenticationButton=document.getElementById("authenticateButton"),personNameInput=document.getElementById("personName"),libraryFaceInput=document.getElementById("libraryFaceImage"),addFaceButton=document.getElementById("addFaceButton"),searchFaceInput=document.getElementById("searchFaceImage"),searchPreview=document.getElementById("searchPreview"),searchPlaceholder=document.getElementById("searchPlaceholder"),startSearchButton=document.getElementById("startSearchButton");
let activeSearchTimer=null;
async function readJsonResponse(response){let responseData={};try{responseData=await response.json();}catch(error){responseData={};}if(response.ok){return responseData;}if(response.status===401){throw new Error("管理员认证无效，请先点击“管理员认证”完成登录");}throw new Error(responseData.message||"请求处理失败");}
function setLibraryStatus(message,isError){const statusElement=document.getElementById("libraryStatus");statusElement.textContent=message;statusElement.className=isError?"status-text error-text":"status-text success-text";}
function formatCreatedTime(createdTime){return createdTime?createdTime.replace("T"," ").slice(0,16):"";}
function createLibraryItem(faceRecord){const item=document.createElement("div"),image=document.createElement("img"),information=document.createElement("div"),name=document.createElement("div"),createdTime=document.createElement("div"),deleteButton=document.createElement("button");item.className="library-item";image.src=faceRecord.image_url;image.alt=faceRecord.person_name;information.className="library-item-info";name.className="library-item-name";name.textContent=faceRecord.person_name;createdTime.className="library-item-time";createdTime.textContent=formatCreatedTime(faceRecord.created_at);deleteButton.className="icon-button";deleteButton.type="button";deleteButton.textContent="×";deleteButton.title="删除该参考照片";deleteButton.addEventListener("click",function(){deleteFaceRecord(faceRecord.id,faceRecord.person_name);});information.append(name,createdTime);item.append(image,information,deleteButton);return item;}
function displayLibraryMessage(message,isError){const libraryList=document.getElementById("libraryList"),emptyElement=document.createElement("div");emptyElement.className=isError?"empty-state error-text":"empty-state";emptyElement.textContent=message;libraryList.replaceChildren(emptyElement);}
function renderFaceLibrary(faceRecords){const libraryList=document.getElementById("libraryList");document.getElementById("libraryCount").textContent=faceRecords.length+" 条参考照片";libraryList.replaceChildren();if(faceRecords.length===0){displayLibraryMessage("人脸库为空，请先录入人员。",false);return;}for(const faceRecord of faceRecords){libraryList.appendChild(createLibraryItem(faceRecord));}}
async function loadFaceLibrary(){try{const response=await fetch("/api/library",{cache:"no-store"});const responseData=await readJsonResponse(response);renderFaceLibrary(responseData.faces);}catch(error){displayLibraryMessage(error.message,true);}}
function openAdministratorAuthentication(){const authenticationWindow=window.open("/admin-auth","faceLibraryAuthentication","width=520,height=420");if(!authenticationWindow){setLibraryStatus("浏览器阻止了认证窗口，请允许弹出窗口",true);return;}setLibraryStatus("请在认证窗口中输入管理员账号和密码",false);}
async function addFaceRecord(){const personName=personNameInput.value.trim(),faceImage=libraryFaceInput.files[0];if(!personName||!faceImage){setLibraryStatus("请输入姓名并选择照片",true);return;}addFaceButton.disabled=true;setLibraryStatus("正在使用 CNN 检测并保存原图与特征…",false);const requestData=new FormData();requestData.append("person_name",personName);requestData.append("face_image",faceImage);try{const response=await fetch("/api/library",{method:"POST",body:requestData,credentials:"same-origin"});await readJsonResponse(response);personNameInput.value="";libraryFaceInput.value="";setLibraryStatus("录入完成",false);await loadFaceLibrary();}catch(error){setLibraryStatus(error.message,true);}finally{addFaceButton.disabled=false;}}
async function deleteFaceRecord(faceRecordId,personName){if(!window.confirm("确定删除“"+personName+"”的这条参考照片吗？")){return;}try{const response=await fetch("/api/library/"+faceRecordId,{method:"DELETE",credentials:"same-origin"});await readJsonResponse(response);setLibraryStatus("删除完成",false);await loadFaceLibrary();}catch(error){setLibraryStatus(error.message,true);}}
function updateSearchPreview(){const selectedFile=searchFaceInput.files[0];if(!selectedFile){searchPreview.style.display="none";searchPlaceholder.style.display="flex";return;}searchPreview.src=URL.createObjectURL(selectedFile);searchPreview.style.display="block";searchPlaceholder.style.display="none";}
function createSearchResultItem(searchResult,resultIndex){const item=document.createElement("div"),image=document.createElement("img"),information=document.createElement("div"),name=document.createElement("div"),rank=document.createElement("div"),similarity=document.createElement("div");item.className="result-item";image.src=searchResult.image_url;image.alt=searchResult.person_name;name.className="result-name";name.textContent=searchResult.person_name;rank.className="result-rank";rank.textContent="当前排名 "+(resultIndex+1);similarity.className="result-similarity";similarity.textContent=Number(searchResult.similarity).toFixed(2)+"%";information.append(name,rank);item.append(image,information,similarity);return item;}
function renderSearchResults(searchResults){const resultsElement=document.getElementById("searchResults");resultsElement.replaceChildren();for(let resultIndex=0;resultIndex<searchResults.length;resultIndex+=1){resultsElement.appendChild(createSearchResultItem(searchResults[resultIndex],resultIndex));}}
function updateSearchProgress(searchStatus){const totalComparisons=searchStatus.total||0,completedComparisons=searchStatus.completed||0,progressPercentage=searchStatus.percentage||0,currentPersonElement=document.getElementById("currentPerson");document.getElementById("progressBar").style.width=progressPercentage+"%";document.getElementById("progressText").textContent=completedComparisons+" / "+totalComparisons;document.getElementById("progressPercentage").textContent=progressPercentage.toFixed(1)+"%";currentPersonElement.textContent=searchStatus.current_person?"正在比对："+searchStatus.current_person:"";renderSearchResults(searchStatus.results||[]);}
async function pollSearchStatus(searchJobId){try{const response=await fetch("/api/search/status/"+searchJobId,{cache:"no-store"});const searchStatus=await readJsonResponse(response);updateSearchProgress(searchStatus);if(searchStatus.status==="completed"){document.getElementById("searchMessage").textContent="搜索完成，用时 "+Number(searchStatus.duration_seconds||0).toFixed(2)+" 秒";document.getElementById("searchMessage").className="status-text success-text";startSearchButton.disabled=false;activeSearchTimer=null;return;}if(searchStatus.status==="error"){throw new Error(searchStatus.message||"搜索任务失败");}document.getElementById("searchMessage").textContent="正在逐条计算相似度…";activeSearchTimer=window.setTimeout(function(){pollSearchStatus(searchJobId);},500);}catch(error){document.getElementById("searchMessage").textContent=error.message;document.getElementById("searchMessage").className="status-text error-text";startSearchButton.disabled=false;activeSearchTimer=null;}}
async function startSequentialSearch(){const searchFaceFile=searchFaceInput.files[0],messageElement=document.getElementById("searchMessage");if(!searchFaceFile){messageElement.textContent="请先选择查询照片";messageElement.className="status-text error-text";return;}if(activeSearchTimer){window.clearTimeout(activeSearchTimer);activeSearchTimer=null;}startSearchButton.disabled=true;messageElement.textContent="正在使用 CNN 检测查询人脸…";messageElement.className="status-text";document.getElementById("progressArea").style.display="block";document.getElementById("progressBar").style.width="0%";document.getElementById("progressText").textContent="0 / 0";document.getElementById("progressPercentage").textContent="0%";document.getElementById("currentPerson").textContent="";document.getElementById("searchResults").replaceChildren();const requestData=new FormData();requestData.append("search_face_image",searchFaceFile);try{const response=await fetch("/api/search/start",{method:"POST",body:requestData});const responseData=await readJsonResponse(response);pollSearchStatus(responseData.search_job_id);}catch(error){messageElement.textContent=error.message;messageElement.className="status-text error-text";startSearchButton.disabled=false;}}
authenticationButton.addEventListener("click",openAdministratorAuthentication);addFaceButton.addEventListener("click",addFaceRecord);searchFaceInput.addEventListener("change",updateSearchPreview);searchPreview.addEventListener("click",function(){searchFaceInput.click();});startSearchButton.addEventListener("click",startSequentialSearch);loadFaceLibrary();
"""

# =============================================================================
# 管理员认证
# =============================================================================

@digest_authentication.get_password
def get_administrator_password(username):
    return ADMINISTRATOR_USERS.get(username)  # 返回 Digest 认证使用的密码

@digest_authentication.error_handler
def handle_authentication_error(status_code=401):
    if request.path.startswith("/api/"):
        return jsonify({"message": "需要管理员 Digest 认证"}), status_code
    return Response("需要管理员 Digest 认证", status=status_code, content_type="text/plain; charset=utf-8")

# =============================================================================
# 目录与数据库
# =============================================================================

def create_required_directories():
    FACE_IMAGE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    TEMPORARY_IMAGE_DIRECTORY.mkdir(parents=True, exist_ok=True)

def get_database_connection():
    database_connection = sqlite3.connect(DATABASE_PATH, timeout=30)  # 每个请求或线程独立连接 SQLite
    database_connection.row_factory = sqlite3.Row
    return database_connection

def initialize_database():
    database_connection = get_database_connection()
    try:
        database_connection.execute("CREATE TABLE IF NOT EXISTS face_library (id INTEGER PRIMARY KEY AUTOINCREMENT, person_name TEXT NOT NULL, image_filename TEXT NOT NULL UNIQUE, encoding_json TEXT NOT NULL, created_at TEXT NOT NULL)")
        database_connection.execute("CREATE INDEX IF NOT EXISTS index_face_library_name ON face_library(person_name)")
        database_connection.commit()
    finally:
        database_connection.close()

def fetch_face_library_records():
    database_connection = get_database_connection()
    try:
        face_records = database_connection.execute("SELECT id, person_name, image_filename, encoding_json, created_at FROM face_library ORDER BY id DESC").fetchall()
    finally:
        database_connection.close()
    return face_records

def fetch_face_search_records():
    database_connection = get_database_connection()
    try:
        face_records = database_connection.execute("SELECT id, person_name, image_filename, encoding_json FROM face_library ORDER BY id ASC").fetchall()
    finally:
        database_connection.close()
    return face_records

def count_face_library_records():
    database_connection = get_database_connection()
    try:
        database_row = database_connection.execute("SELECT COUNT(*) AS record_count FROM face_library").fetchone()
        record_count = int(database_row["record_count"])
    finally:
        database_connection.close()
    return record_count

def insert_face_library_record(person_name, image_filename, face_encoding):
    created_at = time.strftime("%Y-%m-%dT%H:%M:%S")
    encoding_json = json.dumps(face_encoding.tolist())
    database_connection = get_database_connection()
    try:
        database_cursor = database_connection.execute("INSERT INTO face_library(person_name, image_filename, encoding_json, created_at) VALUES (?, ?, ?, ?)", (person_name, image_filename, encoding_json, created_at))
        database_connection.commit()
        face_record_id = int(database_cursor.lastrowid)
    finally:
        database_connection.close()
    return face_record_id, created_at

def delete_face_library_record(face_record_id):
    database_connection = get_database_connection()
    try:
        face_record = database_connection.execute("SELECT image_filename FROM face_library WHERE id = ?", (face_record_id,)).fetchone()
        if face_record is None:
            return None
        image_filename = str(face_record["image_filename"])
        database_connection.execute("DELETE FROM face_library WHERE id = ?", (face_record_id,))
        database_connection.commit()
    finally:
        database_connection.close()
    return image_filename

# =============================================================================
# 原图保存与人脸处理
# =============================================================================

def validate_uploaded_image(uploaded_image):
    if uploaded_image is None or not uploaded_image.filename:
        raise ValueError("请选择需要上传的图片")
    image_extension = Path(uploaded_image.filename).suffix.lower().lstrip(".")
    if image_extension not in ALLOWED_IMAGE_EXTENSIONS:
        raise ValueError("仅支持 JPG、JPEG、PNG 或 WebP 图片")
    return image_extension

def save_uploaded_image_unchanged(uploaded_image, destination_directory, filename_prefix):
    image_extension = validate_uploaded_image(uploaded_image)
    generated_filename = filename_prefix + "_" + uuid.uuid4().hex + "." + image_extension
    generated_path = destination_directory / generated_filename
    uploaded_image.stream.seek(0)
    uploaded_image.save(generated_path)  # 按上传字节原样保存，不缩放、不重编码、不压缩
    try:
        with Image.open(generated_path) as source_image:
            source_image.verify()  # 仅验证图片可读取，不修改原文件
    except (UnidentifiedImageError, OSError, ValueError) as image_error:
        generated_path.unlink(missing_ok=True)
        raise ValueError("图片无法读取或文件内容已损坏") from image_error
    return generated_path, generated_filename

def load_image_array_for_recognition(image_path):
    try:
        with Image.open(image_path) as source_image:
            oriented_image = ImageOps.exif_transpose(source_image)  # 只在内存中修正 EXIF 方向
            rgb_image = oriented_image.convert("RGB")  # 只在内存中转换 RGB，磁盘原图不变
            image_array = np.asarray(rgb_image)
    except (UnidentifiedImageError, OSError, ValueError) as image_error:
        raise ValueError("图片无法读取或文件内容已损坏") from image_error
    return image_array

def extract_single_face_encoding(image_path):
    image_array = load_image_array_for_recognition(image_path)
    face_locations = face_recognition.face_locations(image_array, number_of_times_to_upsample=FACE_LOCATION_UPSAMPLE_COUNT, model=FACE_DETECTION_MODEL)  # 使用 CNN 检测唯一人脸
    if len(face_locations) == 0:
        raise ValueError("图片中未检测到清晰人脸")
    if len(face_locations) > 1:
        raise ValueError("图片中检测到多张人脸，请只保留一张人脸")
    face_encodings = face_recognition.face_encodings(image_array, known_face_locations=face_locations, num_jitters=FACE_ENCODING_JITTER_COUNT, model=FACE_ENCODING_MODEL)  # 使用 large 模型提取 128 维特征
    if not face_encodings:
        raise ValueError("人脸特征提取失败，请更换清晰照片")
    return np.asarray(face_encodings[0], dtype=np.float64)

def convert_distance_to_similarity(face_distance, reference_threshold=0.6):
    numeric_distance = float(face_distance)
    if numeric_distance > reference_threshold:
        similarity_value = (1.0 - numeric_distance) / ((1.0 - reference_threshold) * 2.0)
    else:
        linear_similarity = 1.0 - numeric_distance / (reference_threshold * 2.0)
        nonlinear_adjustment = math.pow((linear_similarity - 0.5) * 2.0, 0.2)
        similarity_value = linear_similarity + (1.0 - linear_similarity) * nonlinear_adjustment
    return round(max(0.0, min(1.0, similarity_value)) * 100.0, 2)

def calculate_face_similarity(first_face_encoding, second_face_encoding):
    face_distances = face_recognition.face_distance([first_face_encoding], second_face_encoding)
    return convert_distance_to_similarity(float(face_distances[0]))

def remove_file_safely(file_path):
    if file_path is not None:
        Path(file_path).unlink(missing_ok=True)

# =============================================================================
# 后台搜索任务
# =============================================================================

def create_search_job(total_comparisons):
    search_job_id = uuid.uuid4().hex
    search_job = {"status": "queued", "message": "", "total": total_comparisons, "completed": 0, "percentage": 0.0, "current_person": "", "results": [], "duration_seconds": 0.0, "created_timestamp": time.time()}
    with SEARCH_JOBS_LOCK:
        SEARCH_JOBS[search_job_id] = search_job
    return search_job_id

def update_search_job(search_job_id, updated_fields):
    with SEARCH_JOBS_LOCK:
        search_job = SEARCH_JOBS.get(search_job_id)
        if search_job is not None:
            search_job.update(updated_fields)

def get_search_job(search_job_id):
    with SEARCH_JOBS_LOCK:
        search_job = SEARCH_JOBS.get(search_job_id)
        if search_job is None:
            return None
        search_job_copy = dict(search_job)
        search_job_copy["results"] = list(search_job["results"])
    return search_job_copy

def prune_expired_search_jobs():
    expiration_timestamp = time.time() - SEARCH_JOB_RETENTION_SECONDS
    expired_search_job_ids = []
    with SEARCH_JOBS_LOCK:
        for search_job_id, search_job in SEARCH_JOBS.items():
            if search_job["created_timestamp"] < expiration_timestamp:
                expired_search_job_ids.append(search_job_id)
        for search_job_id in expired_search_job_ids:
            SEARCH_JOBS.pop(search_job_id, None)

def build_search_result(face_record, similarity_percentage):
    return {"person_name": str(face_record["person_name"]), "similarity": similarity_percentage, "image_url": "/face-image/" + str(face_record["image_filename"])}

def update_person_best_result(best_result_by_person, face_record, similarity_percentage):
    person_name = str(face_record["person_name"])
    previous_result = best_result_by_person.get(person_name)
    if previous_result is not None and float(previous_result["similarity"]) >= similarity_percentage:
        return
    best_result_by_person[person_name] = build_search_result(face_record, similarity_percentage)

def get_search_result_similarity(search_result):
    return float(search_result["similarity"])

def sort_search_results(best_result_by_person):
    search_results = list(best_result_by_person.values())
    search_results.sort(key=get_search_result_similarity, reverse=True)
    return search_results[:SEARCH_RESULT_LIMIT]

def calculate_search_progress(completed_comparisons, total_comparisons):
    if total_comparisons == 0:
        return 100.0
    return round(completed_comparisons / total_comparisons * 100.0, 1)

def process_one_search_record(search_face_encoding, face_record):
    stored_face_encoding = np.asarray(json.loads(str(face_record["encoding_json"])), dtype=np.float64)
    return calculate_face_similarity(stored_face_encoding, search_face_encoding)

def run_sequential_face_search(search_job_id, search_face_encoding):
    search_started_time = time.perf_counter()
    try:
        face_records = fetch_face_search_records()
        total_comparisons = len(face_records)
        best_result_by_person = {}
        update_search_job(search_job_id, {"status": "processing", "total": total_comparisons, "message": ""})
        completed_comparisons = 0
        for face_record in face_records:  # 严格按数据库顺序逐条比较
            person_name = str(face_record["person_name"])
            update_search_job(search_job_id, {"current_person": person_name})
            similarity_percentage = process_one_search_record(search_face_encoding, face_record)
            update_person_best_result(best_result_by_person, face_record, similarity_percentage)
            completed_comparisons += 1
            update_search_job(search_job_id, {"completed": completed_comparisons, "percentage": calculate_search_progress(completed_comparisons, total_comparisons), "results": sort_search_results(best_result_by_person)})
        search_duration = time.perf_counter() - search_started_time
        update_search_job(search_job_id, {"status": "completed", "completed": total_comparisons, "percentage": 100.0, "current_person": "", "duration_seconds": round(search_duration, 3)})
    except Exception as search_error:
        update_search_job(search_job_id, {"status": "error", "current_person": "", "message": "搜索失败：" + str(search_error)})

# =============================================================================
# 页面和 API 路由
# =============================================================================

def render_application_page(page_title, active_page, page_content, page_script):
    compare_class = "active" if active_page == "compare" else ""
    search_class = "active" if active_page == "search" else ""
    navigation_html = '<a class="' + compare_class + '" href="/compare">人脸对比</a><a class="' + search_class + '" href="/search">人脸搜索</a>'
    return render_template_string(BASE_PAGE_TEMPLATE, page_title=page_title, navigation_html=navigation_html, page_content=page_content, page_script=page_script)

def serialize_face_library(face_records):
    serialized_records = []
    for face_record in face_records:
        serialized_records.append({"id": int(face_record["id"]), "person_name": str(face_record["person_name"]), "image_url": "/face-image/" + str(face_record["image_filename"]), "created_at": str(face_record["created_at"])})
    return serialized_records

@application.route("/")
def home_page():
    return redirect(url_for("compare_page"))

@application.route("/compare")
def compare_page():
    return render_application_page("人脸对比", "compare", COMPARE_PAGE_CONTENT, COMPARE_PAGE_SCRIPT)

@application.route("/search")
def search_page():
    return render_application_page("人脸搜索", "search", SEARCH_PAGE_CONTENT, SEARCH_PAGE_SCRIPT)

@application.route("/admin-auth")
@digest_authentication.login_required
def administrator_authentication_page():
    return render_template_string(ADMIN_AUTHENTICATION_PAGE, administrator_username=digest_authentication.username())

@application.route("/face-image/<path:image_filename>")
def face_image(image_filename):
    return send_from_directory(FACE_IMAGE_DIRECTORY, image_filename)

@application.route("/api/compare", methods=["POST"])
def compare_faces_api():
    first_temporary_path = None
    second_temporary_path = None
    try:
        first_temporary_path, _ = save_uploaded_image_unchanged(request.files.get("first_face_image"), TEMPORARY_IMAGE_DIRECTORY, "compare_first")
        second_temporary_path, _ = save_uploaded_image_unchanged(request.files.get("second_face_image"), TEMPORARY_IMAGE_DIRECTORY, "compare_second")
        first_face_encoding = extract_single_face_encoding(first_temporary_path)
        second_face_encoding = extract_single_face_encoding(second_temporary_path)
        return jsonify({"similarity": calculate_face_similarity(first_face_encoding, second_face_encoding)})
    except ValueError as validation_error:
        return jsonify({"message": str(validation_error)}), 400
    except Exception as unexpected_error:
        return jsonify({"message": "对比处理失败：" + str(unexpected_error)}), 500
    finally:
        remove_file_safely(first_temporary_path)
        remove_file_safely(second_temporary_path)

@application.route("/api/library", methods=["GET"])
def list_face_library_api():
    return jsonify({"faces": serialize_face_library(fetch_face_library_records())})

@application.route("/api/library", methods=["POST"])
@digest_authentication.login_required
def add_face_library_api():
    stored_image_path = None
    try:
        person_name = request.form.get("person_name", "").strip()
        if not person_name:
            raise ValueError("请输入人员姓名")
        if len(person_name) > 80:
            raise ValueError("人员姓名不能超过 80 个字符")
        stored_image_path, stored_image_filename = save_uploaded_image_unchanged(request.files.get("face_image"), FACE_IMAGE_DIRECTORY, "library")
        stored_face_encoding = extract_single_face_encoding(stored_image_path)
        face_record_id, created_at = insert_face_library_record(person_name, stored_image_filename, stored_face_encoding)
        return jsonify({"id": face_record_id, "person_name": person_name, "image_url": "/face-image/" + stored_image_filename, "created_at": created_at}), 201
    except ValueError as validation_error:
        remove_file_safely(stored_image_path)
        return jsonify({"message": str(validation_error)}), 400
    except Exception as unexpected_error:
        remove_file_safely(stored_image_path)
        return jsonify({"message": "录入失败：" + str(unexpected_error)}), 500

@application.route("/api/library/<int:face_record_id>", methods=["DELETE"])
@digest_authentication.login_required
def delete_face_library_api(face_record_id):
    image_filename = delete_face_library_record(face_record_id)
    if image_filename is None:
        return jsonify({"message": "指定的人脸记录不存在"}), 404
    remove_file_safely(FACE_IMAGE_DIRECTORY / image_filename)
    return jsonify({"deleted": True})

@application.route("/api/search/start", methods=["POST"])
def start_face_search_api():
    query_temporary_path = None
    try:
        face_library_count = count_face_library_records()
        if face_library_count == 0:
            raise ValueError("人脸库为空，请先录入人员")
        query_temporary_path, _ = save_uploaded_image_unchanged(request.files.get("search_face_image"), TEMPORARY_IMAGE_DIRECTORY, "search_query")
        search_face_encoding = extract_single_face_encoding(query_temporary_path)
        prune_expired_search_jobs()
        search_job_id = create_search_job(face_library_count)
        search_thread = Thread(target=run_sequential_face_search, args=(search_job_id, search_face_encoding.copy()), daemon=True, name="face-search-" + search_job_id[:8])  # 后台线程保证 AJAX 轮询不被长搜索阻塞
        search_thread.start()
        return jsonify({"search_job_id": search_job_id}), 202
    except ValueError as validation_error:
        return jsonify({"message": str(validation_error)}), 400
    except Exception as unexpected_error:
        return jsonify({"message": "无法启动搜索：" + str(unexpected_error)}), 500
    finally:
        remove_file_safely(query_temporary_path)

@application.route("/api/search/status/<search_job_id>", methods=["GET"])
def search_status_api(search_job_id):
    prune_expired_search_jobs()
    search_job = get_search_job(search_job_id)
    if search_job is None:
        return jsonify({"message": "搜索任务不存在或已经过期"}), 404
    response_data = {"status": search_job["status"], "message": search_job["message"], "total": search_job["total"], "completed": search_job["completed"], "percentage": search_job["percentage"], "current_person": search_job["current_person"], "results": search_job["results"], "duration_seconds": search_job["duration_seconds"]}
    return jsonify(response_data)

@application.errorhandler(413)
def request_too_large(_error):
    if request.path.startswith("/api/"):
        return jsonify({"message": "上传文件过大，单次请求不能超过 64MB"}), 413
    return "上传文件过大，单次请求不能超过 64MB", 413

# =============================================================================
# 启动
# =============================================================================

create_required_directories()
initialize_database()

if __name__ == "__main__":
    application.run(host="0.0.0.0", port=8080, debug=False, threaded=True)  # 开发服务器；生产环境应使用 WSGI 和 HTTPS
