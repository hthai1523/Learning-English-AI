📘 teacherAI — Ứng dụng Học Tiếng Anh Tương Tác

teacherAI là một ứng dụng học tiếng Anh theo phong cách cá nhân hoá – trực quan – trò chuyện thời gian thực, gồm hai phần:

Backend (FastAPI + Python)

Frontend (React + Vite)

Ứng dụng dành cho người dùng cuối và cả người muốn học, nhưng cũng cực kỳ dễ để developer non-coder tải về và chạy bằng Claude Code.

🚀 TÍNH NĂNG CHÍNH
🗣 Live Talk – Hội thoại thời gian thực

Nhắn tin hoặc nói chuyện trực tiếp với “giáo viên AI”.

🧠 Lesson AI – Bài học theo cấp độ

Bao gồm: từ vựng, luyện phản xạ, hội thoại, trắc nghiệm, luyện nói.

👩‍🏫 Avatar 3D

Hiển thị giáo viên 3D sinh động bằng file .glb.

🎧 TTS – Text to Speech

Backend phát âm câu thoại, từ vựng, luyện nói.

📈 Dashboard

Theo dõi tiến độ học, điểm nói, điểm phát âm.

📦 CẤU TRÚC THƯ MỤC REPO
teacherAI/
│
├── backend/                # FastAPI backend
│   ├── main.py             # Điểm vào của API
│   ├── routers/            # API routes
│   ├── services/           # Logic xử lý
│   ├── models/             # Schemas
│   ├── media/tts/          # File âm thanh tạo ra
│   ├── config.py
│   ├── requirements.txt
│   └── .env.example        # File mẫu (không có API key)
│
├── frontend/               # React frontend
│   ├── src/
│   ├── public/
│   ├── package.json
│   └── vite.config.js
│
├── .gitignore
└── README.md

🔐 BẢO MẬT — CỰC QUAN TRỌNG (DÀNH CHO NGƯỜI TẢI VỀ)

Ứng dụng không chứa API key thật.
Để chạy backend, bạn phải tự tạo file .env.

Trong thư mục backend/, tạo file:

.env


Nội dung:

OPENAI_API_KEY=YOUR_OPENAI_KEY_HERE


➡️ Tuyệt đối không commit file .env lên GitHub.
➡️ .gitignore đã chặn sẵn để bảo vệ bạn.

🛠 HƯỚNG DẪN CHẠY ỨNG DỤNG — DÀNH CHO NON-CODER

⚡️ CÁCH 1 — Dành cho người dùng Claude Code (DỄ NHẤT)

Trong Claude Code → mở repo → bấm:

Run Backend


Nếu Claude tự detect chuẩn, nó sẽ chạy FastAPI bằng:

uvicorn main:app --reload --host 0.0.0.0 --port 8000


Luego:

Run Frontend


Claude sẽ dùng:

npm install
npm run dev


Website chạy tại:

👉 http://localhost:5173

Backend chạy tại:

👉 http://localhost:8000

Hết. Xong.

⚡️ CÁCH 2 — Chạy thủ công trên máy ( dành cho người không dùng Claude )
1️⃣ Cài Python + NodeJS

Python 3.10+

Node 18+

2️⃣ Chạy Backend (Python)

Mở terminal:

cd backend


Cài thư viện:

pip install -r requirements.txt


Tạo file .env:

cp .env.example .env


Mở .env → thêm OpenAI API key của bạn.

Chạy server:

uvicorn main:app --reload --port 8000


🟩 Backend chạy tại:
http://localhost:8000/docs

3️⃣ Chạy Frontend (React)

Mở terminal:

cd frontend
npm install
npm run dev


🟦 Frontend chạy tại:
http://localhost:5173

📡 KẾT NỐI FRONTEND → BACKEND

Frontend dùng API:

http://localhost:8000


Không cần chỉnh gì thêm.

🧩 CÁC VẤN ĐỀ THƯỜNG GẶP (DÀNH CHO NON-CODER)
❌ Backend báo "Key not found"

→ Bạn chưa tạo file .env.

❌ “Module not found" khi chạy npm install

→ Cập nhật NodeJS lên 18+.

❌ Avatar 3D bị lỗi hiển thị

→ File .glb > 50MB nhưng vẫn hoạt động bình thường.
