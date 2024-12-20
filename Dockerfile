# # The builder image, used to build the virtual environment
# FROM python:3.11-buster as builder

# RUN apt-get update && apt-get install -y git

# RUN pip install poetry==1.4.2

# ENV POETRY_NO_INTERACTION=1 \
#     POETRY_VIRTUALENVS_IN_PROJECT=1 \
#     POETRY_VIRTUALENVS_CREATE=1 \
#     POETRY_CACHE_DIR=/tmp/poetry_cache

# WORKDIR /app

# COPY pyproject.toml poetry.lock ./

# RUN poetry install --without dev --no-root && rm -rf $POETRY_CACHE_DIR

# # The runtime image, used to just run the code
# FROM python:3.11-slim-buster as runtime

# WORKDIR /app

# ENV VIRTUAL_ENV=/app/.venv \
#     PATH="/app/.venv/bin:$PATH"

# COPY --from=builder ${VIRTUAL_ENV} ${VIRTUAL_ENV}

# # Copy only necessary files and directories
# COPY app.py ./
# COPY streamlit_agent ./streamlit_agent
# COPY config ./config

# CMD ["streamlit", "run", "app.py", "--server.port", "8051"]

# The builder image, used to build the virtual environment
FROM python:3.11-buster as builder

# Установить системные зависимости
RUN apt-get update && apt-get install -y git

# Установить Poetry
RUN pip install poetry==1.4.2

# Установить переменные среды для Poetry
ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

# Рабочая директория
WORKDIR /app

# Скопировать только pyproject.toml, так как poetry.lock еще нет
COPY pyproject.toml ./

# Сгенерировать poetry.lock
RUN poetry lock

# Установить зависимости (без dev-зависимостей)
RUN poetry install --without dev --no-root && rm -rf $POETRY_CACHE_DIR

# The runtime image, used to just run the code
FROM python:3.11-slim-buster as runtime

# Рабочая директория
WORKDIR /app

# Настроить виртуальное окружение
ENV VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

# Скопировать готовое окружение из builder-слоя
COPY --from=builder ${VIRTUAL_ENV} ${VIRTUAL_ENV}

# Копировать только необходимые файлы для запуска приложения
COPY app.py ./
COPY streamlit_agent ./streamlit_agent
COPY config ./config
COPY models.yml ./models.yml
# Запуск приложения
CMD ["streamlit", "run", "app.py", "--server.port", "8051"]
