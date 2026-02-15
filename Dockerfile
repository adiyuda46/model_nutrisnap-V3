# Gunakan image Python 3.9 sebagai base
FROM python:3.9-slim

# Set environment variable
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Buat direktori untuk aplikasi
WORKDIR /app

# Salin file requirements.txt dan install dependensi
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Salin semua file aplikasi
COPY . .

# Jalankan aplikasi
CMD ["python", "main.py"]
