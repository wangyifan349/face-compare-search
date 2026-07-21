# 👤 Face Compare and Search

Face Compare and Search is a small Flask application that provides two browser pages in one standalone Python file: `/compare` compares two face images and displays a similarity score, while `/search` manages a local face library and searches one uploaded face against every stored encoding with AJAX progress updates.

This repository provides two complete editions of the same application. `face_compare_search_zh.py` contains the full Chinese interface and Chinese comments. `face_compare_search_en.py` contains the full English interface and English comments. Each file includes its own HTML, CSS, JavaScript, Flask routes, SQLite logic, HTTP Digest Authentication, image handling, face enrollment, comparison, and sequential search. No template, static, configuration, or language directories are required.

## 🚀 Download, Install, and Run

Run these commands in order:

```bash
git clone https://github.com/wangyifan349/face-compare-search.git
cd face-compare-search
pip install flask flask-httpauth pillow numpy face-recognition
python -c "import dlib, face_recognition; print('Dependencies are ready')"
# To use the English edition instead, stop the Chinese edition and run:
python face_compare_search_en.py

python -c "import dlib; print('===== dlib CUDA Detection ====='); print('CUDA Support:', dlib.DLIB_USE_CUDA); gpu_count = dlib.cuda.get_num_devices() if dlib.DLIB_USE_CUDA else 0; print('GPU Count:', gpu_count); [print('GPU %d:' % i, dlib.cuda.get_device_name(i)) for i in range(gpu_count)] if dlib.DLIB_USE_CUDA else None"


# Start the Chinese edition
python face_compare_search_zh.py
```

After the server starts, open:

```text
http://127.0.0.1:8080/compare
http://127.0.0.1:8080/search
```

⚠️ Run only one edition at a time. Both files listen on port `8080` and use the same `face_data` directory.

`face-recognition` depends on dlib. CNN detection may be slow on CPU, especially with large original images. A CUDA-enabled dlib build and a compatible NVIDIA GPU can significantly improve detection speed.

## ✨ Main Functions

* 🔍 Compare two uploaded images and display only their face similarity.
* 📥 Enroll a person name and an original reference image into SQLite.
* 🧑‍💻 Search a query image against stored face encodings one record at a time.
* 📊 Display current progress, the person currently being compared, elapsed time, and ranked results through AJAX polling.
* 🏆 Keep the highest result when one person has multiple reference images.
* 🔐 Protect face enrollment and record deletion with HTTP Digest Authentication.
* 🖼️ Preserve uploaded image files without resizing, cropping, format conversion, re-encoding, or additional compression.

## 🧠 Models and Image Processing

The application uses the following configuration:

```python
FACE_DETECTION_MODEL = "cnn"
FACE_ENCODING_MODEL = "large"
FACE_LOCATION_UPSAMPLE_COUNT = 1
FACE_ENCODING_JITTER_COUNT = 1
```

`cnn` is used to locate faces. `large` is passed to `face_recognition.face_encodings` to use the larger landmark model before generating the 128-dimensional encoding. Every uploaded image must contain exactly one detectable face.

Uploaded files are saved with their original bytes and original extension. Pillow is used only to verify that the saved file is readable. During recognition, EXIF orientation correction and RGB conversion occur in memory; the stored file is not rewritten. Supported extensions are JPG, JPEG, PNG, and WebP. The maximum request size is 64 MB.

Search does not run CNN detection again for every stored library image. CNN detection and face encoding are performed when a reference image is enrolled and when a query image is uploaded. The search loop compares the already stored 128-dimensional encodings.

## 🔐 Administrator Authentication

The default administrator accounts are defined near the top of both files:

```python
ADMINISTRATOR_USERS = {
    "admin": "FaceAdmin@2026",
    "operator": "FaceOperator@2026"
}
```

⚠️ Change these passwords and `application.config["SECRET_KEY"]` before making the service publicly accessible.

On the search page, click the administrator-login button. The browser opens `/admin-auth` and requests HTTP Digest credentials. Successful authentication allows the same browser to enroll and delete face records. Comparison, search, library metadata, reference-image delivery, and search-status routes remain public in the current version.

HTTP Digest Authentication does not replace HTTPS. Use HTTPS when the application is accessible over a network or stores real biometric data.

## 💾 Data Storage

The application creates the following structure beside the Python file:

```text
face_data/
├── face_images/
├── temporary_images/
└── face_library.sqlite3
```

`face_images` stores enrolled original reference images. `temporary_images` temporarily stores comparison and query uploads and removes them after processing. `face_library.sqlite3` stores person names, image filenames, JSON-encoded face vectors, and creation times.

The database table is:

```sql
CREATE TABLE face_library (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    person_name TEXT NOT NULL,
    image_filename TEXT NOT NULL UNIQUE,
    encoding_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
```

The schema is compatible with the earlier single-file version. To reuse existing data, place the new Python file beside the existing `face_data` directory. Back up the complete `face_data` directory because the database and reference images must remain together.

## 🌐 Routes and APIs

| Method | Route                                | Authentication | Purpose                                               |
| ------ | ------------------------------------ | -------------- | ----------------------------------------------------- |
| GET    | `/`                                  | No             | Redirect to `/compare`                                |
| GET    | `/compare`                           | No             | Display the face comparison page                      |
| GET    | `/search`                            | No             | Display face-library management and sequential search |
| GET    | `/admin-auth`                        | Digest         | Request administrator authentication                  |
| GET    | `/face-image/<image_filename>`       | No             | Return an enrolled reference image                    |
| POST   | `/api/compare`                       | No             | Compare `first_face_image` and `second_face_image`    |
| GET    | `/api/library`                       | No             | Return all face-library records                       |
| POST   | `/api/library`                       | Digest         | Enroll `person_name` and `face_image`                 |
| DELETE | `/api/library/<face_record_id>`      | Digest         | Delete a record and its reference image               |
| POST   | `/api/search/start`                  | No             | Start a search using `search_face_image`              |
| GET    | `/api/search/status/<search_job_id>` | No             | Return search progress and results                    |

A successful comparison returns:

```json
{"similarity": 91.27}
```

Starting a search returns HTTP `202`:

```json
{"search_job_id": "job-id"}
```

A search-status response has this structure:

```json
{
  "status": "processing",
  "message": "",
  "total": 100,
  "completed": 38,
  "percentage": 38.0,
  "current_person": "Example",
  "results": [
    {
      "person_name": "Example",
      "similarity": 92.41,
      "image_url": "/face-image/library_xxx.jpg"
    }
  ],
  "duration_seconds": 0.0
}
```

Search states are `queued`, `processing`, `completed`, and `error`. The browser requests status every 500 milliseconds. Search jobs are stored in memory and removed after one hour.

## 🔎 Search Behavior and Limitations

The background thread reads a stable database snapshot in ascending record-ID order. For each record, it loads the stored encoding, calculates Euclidean face distance, converts that distance into a display similarity, updates progress, and sorts the current results. The page displays up to 20 people. When one person has several reference images, only that person's highest similarity is retained.

The displayed similarity is intended for ranking and manual review. It is not a probability, a legal identity determination, or a guaranteed match result. Lighting, pose, age, expression, obstruction, image quality, and camera distortion may affect the value. This project does not include liveness detection and should not be the only control for payments, critical access control, or other high-risk identity verification.

The Flask development server is suitable for local use and testing. Search-job state exists only inside one Python process. A multi-process deployment requires shared job storage such as Redis or a database-backed task system; otherwise a status request may reach a different process and fail to find the job.

## 🛠️ Common Problems

**🐢 CNN processing is slow:** this is expected on CPU and with high-resolution images. Use CUDA-enabled dlib and a compatible NVIDIA GPU for better performance.

**🚫 No face is detected:** use a clear image containing one sufficiently large, visible face.

**👥 Multiple faces are detected:** use an image containing only one person.

**🔒 Enrollment returns HTTP 401:** open `/admin-auth`, complete Digest authentication, and submit again.

**🪟 The authentication window does not open:** allow pop-ups for the application address.

**🔌 Port 8080 is already in use:** stop the other process or change the `port=8080` value near the bottom of the selected Python file.

**📂 Existing records are missing:** confirm that the expected `face_data` directory is beside the Python file being executed.

## 📜 License

This project is licensed under the GNU Affero General Public License version 3.0 only (`AGPL-3.0-only`). The complete license text is provided in [LICENSE](LICENSE).

You may use, study, modify, and redistribute the program under the license terms. If you distribute modified versions, you must preserve the applicable notices and provide the corresponding source as required by the license. If you modify the program and make that modified version available to users over a computer network, GNU AGPLv3 requires that those remote users be offered access to the corresponding source of the version running on the server.

## 💛 Acknowledgements

This project would not be possible without the work of the open-source community. We sincerely thank [Flask](https://flask.palletsprojects.com/) for providing the lightweight web framework, routing, request handling, and application foundation; [Flask-HTTPAuth](https://flask-httpauth.readthedocs.io/) for providing HTTP Digest Authentication used to protect face enrollment and deletion operations; [face_recognition](https://face-recognition.readthedocs.io/) for providing the high-level APIs used to detect faces, generate 128-dimensional face encodings, calculate face distances, and compare facial features; [dlib](https://dlib.net/) for providing the underlying machine-learning models, CNN face detector, facial landmark processing, and face-recognition capabilities used by `face_recognition`; [Pillow](https://pillow.readthedocs.io/) for validating uploaded images, reading image files, handling EXIF orientation, and converting images into the RGB format required for recognition; [NumPy](https://numpy.org/) for efficient numerical arrays and face-vector processing; and [SQLite](https://www.sqlite.org/) for providing a lightweight, reliable, and serverless database for storing names, image filenames, face encodings, and creation times.

🌍 We are deeply grateful to the developers, maintainers, contributors, documentation writers, testers, users, and the wider open-source community for sharing their knowledge and software so that projects such as Face Compare and Search can be developed, studied, improved, and shared with others.
