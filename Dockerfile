# 多阶段构建：先在 build 阶段装依赖，再拷到瘦身运行镜像
FROM python:3.11-slim AS build
WORKDIR /build
RUN pip install --no-cache-dir --upgrade pip
COPY pyproject.toml ./
# 仅装运行时依赖到独立目录
RUN pip install --no-cache-dir --prefix=/install .

FROM python:3.11-slim
WORKDIR /app
# 拷贝安装好的包
COPY --from=build /install /usr/local
COPY app ./app
COPY .env.example ./.env

ENV PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
