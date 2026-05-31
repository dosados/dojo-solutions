"""Метаданные self/dev: какие статьи исключать из локальной оценки."""

from __future__ import annotations

# article-4 — мемуары о Пушкине; архаичный ref, не совпадает с современным переводом модели.
PUSHKIN_ARTICLE = "article-4"

# Статьи, которые пропускаем в dev_it (IT/новости без Пушкина).
DEV_IT_EXCLUDE: tuple[str, ...] = (PUSHKIN_ARTICLE,)
