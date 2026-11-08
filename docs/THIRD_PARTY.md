# Сторонний публичный компонент

## Источник и атрибуция

Финальный ensemble использует один внешний компонент из публичного решения команды **Anti7**:

- репозиторий: https://github.com/Velsaf-dev/molecular-ap-model;
- pinned commit: `2697ab703963d34c748912048134b8b725633cd7`;
- исходный файл: `experiments/exp_017_pure_ml_ultra_sota/submission.csv`;
- raw URL: https://raw.githubusercontent.com/Velsaf-dev/molecular-ap-model/2697ab703963d34c748912048134b8b725633cd7/experiments/exp_017_pure_ml_ultra_sota/submission.csv;
- SHA-256: `d88cd19486c23b3ba44c73f898ac6dba7731b252ec35fefe3806435d912ca99b`.

Локальная копия: `data/public_model_exp017.csv`. Скрипт `scripts/download_public_model.py` проверяет схему, 597 строк, конечные неотрицательные значения и SHA-256. Опция `--refresh` повторно загружает файл именно из pinned commit.

## Как компонент используется

```text
SI_final = 0.5 × SI_BSC40_retrained + 0.5 × SI_Anti7_public_exp017
```

Собственный BSC-40 обучается заново в notebook. Публичный компонент Anti7 используется на этапе ensemble как готовая модельная часть pipeline и не добавляется в train в качестве экспериментальной SI-разметки.

Состав материалов:

- `reproducible_public_sar_5050.ipynb` — основной финальный ensemble;
- `reproducible_bsc40_public_sar_direct.ipynb` — дополнительный эксперимент без публичного model component, с обучением собственных моделей и явно перечисленными structural SAR-правилами.

Официальные Kaggle MSE этих двух версий различаются и не переносятся одна на другую: 72 155.31593 и 82 826.34914 соответственно, по сообщениям пользователя после загрузок.

## Лицензия

В исходном публичном репозитории на зафиксированном commit отдельный `LICENSE`/`COPYING` не обнаружен. Поэтому MIT-лицензия данного репозитория не распространяется на сторонний компонент и не изменяет права его авторов. Здесь сохранены ссылка, атрибуция, точная версия и контрольная сумма. Сторонний training/inference pipeline целиком не копируется; используемые в from-scratch альтернативе structural SAR-правила перечислены заново с явной атрибутацией источника.
