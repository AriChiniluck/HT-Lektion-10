# Пояснення системи тестування мультиагентного RAG-пайплайну

> Цей документ — для тих, хто ніколи раніше не писав автоматизованих тестів для агентів і хоче зрозуміти, **що**, **як** і **чому** тут тестується.

---

## Загальна ідея

Наша система — це мультиагентний пайплайн:

```
User → Supervisor → Planner → Researcher → Critic → (REVISE ↩ / APPROVE) → save_report
```

Кожен агент — окремий LLM-промпт з інструментами. Якість роботи не можна перевірити простим `assert result == "expected"`, бо мова природна й відповіді щоразу різні.

**Рішення:** використовуємо [DeepEval](https://github.com/confident-ai/deepeval) — фреймворк для оцінки LLM-виходів через:
- **метрики**: числа від 0 до 1, де 1 = ідеально
- **пороги (threshold)**: мінімально прийнятна оцінка
- **GEval**: LLM сам оцінює якість відповіді по кроках (evaluation steps)

---

## Структура тестів

```
tests/
├── conftest.py              ← спільна конфігурація pytest
├── golden_dataset.json      ← 15 еталонних прикладів (gold standard)
├── test_planner.py          ← тести агента-планувальника
├── test_researcher.py       ← тести агента-дослідника
├── test_critic.py           ← тести агента-критика
├── test_tools.py            ← тести правильності виклику інструментів
└── test_e2e.py              ← наскрізні тести всього пайплайну
```

---

## Як запустити

```bash
# Всі тести (займає ~20–30 хв)
deepeval test run tests/

# З детальним debug-виводом (показує кожен tool call, вердикт, результат)
DEBUG=1 deepeval test run tests/
deepeval test run tests/ --agent-debug

# Один файл
deepeval test run tests/test_critic.py

# Тільки конкретний тест
deepeval test run tests/test_e2e.py -k happy

# або
deepeval test run tests/test_e2e.py -k full

# Щоб спочатку лише перевірити колекцію тестів (швидко, без LLM-викликів)
deepeval test run tests/ --collect-only

# стандартна поведінка pytest: крапка . = тест пройшов, без деталей. Щоб бачити повний вивід: 
deepeval test run tests/ -v -s
    -v (verbose) — замість . показує повну назву кожного тесту та PASSED / FAILED
    -s (no capture) — не приховує print() та debug_print() виклики всередині тестів

# --agent-debug:
deepeval test run tests/ -v -s --agent-debug

# тільки failures детально, але без зайвого шуму
deepeval test run tests/ -v --tb=short

# Прогрес-бар з %% — це deepeval's rich-рендер, який в PowerShell іноді не відображається коректно. Щоб стабілізувати, встановіть перед запуском:
$env:FORCE_COLOR = "1"
deepeval test run tests/ -v -s


# Один тест — через :: після імені файлу
deepeval test run tests/test_planner.py::test_planner_calls_knowledge_search

# Або через фільтр -k (по частині назви)
deepeval test run tests/ -k "knowledge_search"

# Кілька тестів, — через -k з OR
deepeval test run tests/ -k "knowledge_search or groundedness or save_report"

# Лише конкретний файл
deepeval test run tests/test_planner.py

```

---

## 1. `golden_dataset.json` — еталонний датасет

**Що це?**
Файл з 15 прикладами формату `input` → `expected_output` + `category`.

```json
{
  "input": "Compare naive RAG vs sentence-window retrieval",
  "expected_output": "Naive RAG splits documents into fixed-size chunks...",
  "category": "happy_path"
}
```

**Категорії:**

| Категорія | Кількість | Призначення |
|---|---|---|
| `happy_path` | 5 | Типові запити — система має відповісти повно та точно |
| `edge_case` | 5 | Неоднозначні або занадто широкі запити |
| `failure_case` | 5 | Запити поза доменом — система має відмовити коректно |

**Навіщо?**
Щоразу запускаючи `deepeval test run tests/test_e2e.py`, ми перевіряємо: чи не «зрегресував» пайплайн — тобто чи не погіршилася якість порівняно з еталоном.

---

## 2. `conftest.py` — спільна конфігурація

```python
# tests/conftest.py

def pytest_addoption(parser):
    parser.addoption("--agent-debug", action="store_true")

@pytest.fixture(autouse=True, scope="session")
def configure_debug(request):
    # Якщо запущено з --agent-debug або DEBUG=1 — вмикаємо debug mode
    if request.config.getoption("--agent-debug") or os.getenv("DEBUG") == "1":
        settings.debug = True
    # Передаємо ключ OpenAI в deepeval
    os.environ.setdefault("OPENAI_API_KEY", settings.openai_api_key.get_secret_value())
```

**Що тут відбувається:**
- `autouse=True, scope="session"` — ця фікстура виконується автоматично **один раз** перед усією сесією тестів.
- `settings.debug = True` — вмикає ті самі `debug_print()` виклики, що й команда `debug on` в інтерактивному режимі. Тому в тестах ви бачите ті самі повідомлення: `🔧 web_search(...)`, `📎 [4 documents found]` тощо.

---

## 3. `test_planner.py` — тест Planner-агента

**Що тестується:**
- Чи планувальник будує **конкретні** пошукові запити (не розмиті)?
- Чи вказує правильні джерела (`knowledge_base`, `web`)?
- Чи зберігає мову користувача (якщо питання українською — план теж українською)?

### Метрика: GEval "Plan Quality"

```python
plan_quality = GEval(
    name="Plan Quality",
    evaluation_steps=[
        "Check that the plan contains specific search queries (not vague)",
        "Check that sources_to_check includes relevant sources for the topic",
        "Check that the output_format matches what the user asked for",
        "Check that the goal clearly captures the user's intent",
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    model=EVAL_MODEL,
    threshold=0.7,  # мінімум 70% якості плану
)
```

**Як це працює:** DeepEval надсилає в LLM запит: «Оціни відповідь по цих критеріях від 0 до 1». Це і є GEval.

### Приклад тесту:

```python
@pytest.mark.parametrize("user_input", [
    "Compare naive RAG vs sentence-window retrieval",
    "What are the best practices for multi-agent systems in 2026?",
])
def test_planner_plan_quality(user_input):
    actual_output = plan.invoke({"request": user_input})  # викликаємо реальний агент
    test_case = LLMTestCase(input=user_input, actual_output=actual_output)
    deepeval.assert_test(test_case, [plan_quality])  # оцінюємо та виконуємо assert
```

**Debug-вивід:**
```
[test_planner] → plan(request='Compare naive RAG...')
[test_planner] ← plan returned (2432 chars):
{
  "goal": "Compare naive RAG vs sentence-window retrieval...",
  "search_queries": ["\"naive RAG\" definition chunking...", ...]
}
[test_planner] Running GEval 'Plan Quality' on: 'Compare naive RAG...'
. (PASSED: 1.0)
```

---

## 4. `test_researcher.py` — тест Researcher-агента

**Що тестується:**
- Чи відповідь агента базується на знайдених джерелах (**Groundedness**)?
- Чи відповідь стосується плану дослідження (**Research Relevancy**)?

### Метрика: GEval "Groundedness"

```python
groundedness = GEval(
    name="Groundedness",
    evaluation_steps=[
        "Extract factual claims from 'actual output'.",
        "For each claim, check if it is supported by 'retrieval context'.",
        "Claims consistent with context count as grounded.",
        "Score = grounded / total claims.",
    ],
    evaluation_params=[
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.RETRIEVAL_CONTEXT,  # те, що знайшли інструменти
    ],
    threshold=0.4,  # 40% — порог для гібридного KB+web пошуку
)
```

> **Чому порог 0.4, а не 0.7?**
> Researcher використовує і локальну базу знань (`knowledge_search`), і веб (`web_search`). Deepeval бачить лише короткі витяги з ToolMessage — не повні документи. Частина правильних тверджень базується на веб-даних, які не вкладаються у 2500 символів контексту. Тому 0.4 — чесний реалістичний поріг для нашої гібридної системи.

**Як збираються retrieval_contexts:**

```python
def _run_researcher(plan_text):
    agent = get_research_agent()
    result = agent.invoke({"messages": [{"role": "user", "content": plan_text}]})
    messages = result.get("messages", [])

    retrieval_contexts = []
    for msg in messages:
        if isinstance(msg, ToolMessage):  # результат інструменту
            content = getattr(msg, "content", "")
            if content.strip():
                retrieval_contexts.append(content[:2500])  # перші 2500 символів

    # Остання AIMessage без tool_calls = фінальна відповідь
    final_output = ...
    return final_output, retrieval_contexts
```

---

## 5. `test_critic.py` — тест Critic-агента

**Що тестується:**
- Чи критика конкретна (не розмита)?
- Чи `revision_requests` дієвий (дослідник може по ньому діяти)?
- Чи verdict (APPROVE/REVISE) консистентний з іншими полями (`is_complete`, `is_fresh`)?

### Метрика 1: GEval "Critique Quality"

```python
critique_quality = GEval(
    name="Critique Quality",
    evaluation_steps=[
        "Check that the critique identifies specific issues, not vague complaints.",
        "Check that revision_requests are actionable.",
        "If verdict is APPROVE, gaps should be empty or minor.",
        "If verdict is REVISE, there must be at least one revision_request.",
        "Verify freshness, completeness, and structure are evaluated separately.",
    ],
    threshold=0.7,
)
```

### Метрика 2 (кастомна): GEval "Verdict Consistency"

```python
verdict_consistency = GEval(
    name="Verdict Consistency",
    evaluation_steps=[
        "If verdict is APPROVE: is_complete, is_fresh, is_well_structured should be True.",
        "If verdict is REVISE: at least one boolean should be False AND revision_requests non-empty.",
        "Penalise contradictions (e.g. APPROVE with is_complete=False).",
    ],
    threshold=0.6,
)
```

> Це **кастомна бізнес-логіка**: ми самі визначаємо, що вважається правильним вердиктом Критика, і GEval перевіряє, чи система дотримується цього контракту.

**Приклад тесту:**

```python
def test_critic_revises_weak_findings():
    result = critique.invoke({
        "original_request": RESEARCH_REQUEST,
        "findings": "RAG is good. There are different types. Some are better than others.",
    })
    payload = json.loads(result)
    assert payload["verdict"] == "REVISE"            # ← прямий assert
    assert payload["revision_requests"]              # ← є хоч один запит на виправлення
    # + deepeval перевіряє якість самої критики
    deepeval.assert_test(test_case, [critique_quality])
```

---

## 6. `test_tools.py` — тести правильності інструментів

**Що тестується:**
Чи агенти викликають **правильні** інструменти для правильних ситуацій?

```python
TOOL_METRIC = ToolCorrectnessMetric(threshold=0.5, model=EVAL_MODEL)
```

| Тест | Хто? | Очікуваний виклик |
|---|---|---|
| Test 1 | Planner | `knowledge_search` (для RAG-теми з курсу) |
| Test 2 | Researcher | `knowledge_search`, `web_search` або `read_url` |
| Test 3 | Supervisor | `plan` → `research` → `critique` → `save_report` |

**Як збираємо tool calls:**

```python
def _extract_ai_tool_calls(messages):
    captured = []
    for msg in messages:
        if isinstance(msg, AIMessage):
            for tc in (getattr(msg, "tool_calls", []) or []):
                name = tc.get("name", "")
                args = tc.get("args", {})
                if name:
                    captured.append(ToolCall(name=name, input_parameters=args))
    return captured
```

**Автоматичне підтвердження save_report:**

```python
for round_num in range(max_rounds):
    tool_calls, interrupts, _ = _stream_supervisor_collect(payload, config)
    all_tool_calls.extend(tool_calls)

    if not interrupts:
        break  # пайплайн завершився без перерви

    # HITL interrupt від save_report → автоматично підтверджуємо
    payload = Command(resume={"decisions": [{"type": "approve"}]})
```

---

## 7. `test_e2e.py` — наскрізні тести

**Що тестується:**
Весь пайплайн від запиту користувача до збереженого звіту.

### Метрика 1: AnswerRelevancyMetric

```python
answer_relevancy = AnswerRelevancyMetric(threshold=0.7, model=EVAL_MODEL)
```
Перевіряє: чи відповідь стосується питання? (Без urмошних витягів — просто: «текст по темі?»)

### Метрика 2: GEval "Correctness"

```python
correctness = GEval(
    name="Correctness",
    evaluation_steps=[
        "Check whether facts in 'actual output' contradict 'expected output'.",
        "Penalise omission of critical details.",
        "Different wording of the same concept is acceptable.",
    ],
    evaluation_params=[INPUT, ACTUAL_OUTPUT, EXPECTED_OUTPUT],
    threshold=0.6,
)
```

### Тест на happy_path (3 приклади):
```python
@pytest.mark.parametrize("example", _load_golden("happy_path")[:3])
def test_e2e_happy_path(example):
    actual_output = run_pipeline(example["input"])      # запускаємо весь пайплайн
    test_case = LLMTestCase(
        input=example["input"],
        actual_output=actual_output,
        expected_output=example["expected_output"],
    )
    deepeval.assert_test(test_case, [answer_relevancy, correctness])
```

### Тест на весь датасет (aggregate):
```python
def test_e2e_full_golden_dataset():
    # Запускаємо пайплайн для всіх 15 прикладів
    # НЕ робимо assert на кожному — збираємо статистику
    passed = 0
    for example in dataset:
        ...
        for metric in [answer_relevancy, correctness]:
            metric.measure(test_case)
            if metric.success:
                passed += 1

    pass_rate = passed / total
    assert pass_rate >= 0.60  # мінімум 60% прикладів мають пройти
```

> Чому 60%? Це **базовий поріг** (baseline). Цей числовий показник слід підвищувати поступово, після кожного покращення системи. Починати з 0.95 з першого дня — нереалістично.

---

## 8. Які моделі використовуються?

### Моделі виконання (Execution) — агенти

Всі агенти використовують `settings.model_name` з файлу `.env`:

| Агент | Використовує |
|---|---|
| Planner | `settings.model_name` (наприклад, `gpt-5.1`) |
| Researcher | те саме |
| Critic | те саме |
| Supervisor | те саме |

Ці виклики **коштують найбільше** — агенти використовують багато токенів для планування, пошуку та синтезу.

### Модель оцінювання (Evaluation) — DeepEval GEval

За замовчуванням: та сама `settings.model_name`. Але це **надмірно**.

```python
# У кожному тестовому файлі:
EVAL_MODEL = os.getenv("DEEPEVAL_MODEL", settings.model_name)
```

**GEval лише порівнює тексти** — він не пише звіти і не шукає інформацію. Для цього достатньо меншої моделі.

### Рекомендація: розділити моделі

```bash
# Виконання: gpt-5.1 (потужна, для якісних звітів)
# Оцінювання: gpt-4o-mini (швидша, дешевша, достатня для judge)

DEEPEVAL_MODEL=gpt-4o-mini deepeval test run tests/
# або
set DEEPEVAL_MODEL=gpt-4o-mini
deepeval test run tests/
```

**Переваги:**
- Економія ~60–80% на вартості тестування
- Швидший запуск тестів (mini-моделі швидше відповідають)
- Якість оцінки практично не падає — GEval добре працює з меншими моделями

**Де змінити:**
Додайте у `.env`:
```
DEEPEVAL_MODEL=gpt-4o-mini
```
або використовуйте окремо для тестування, щоб не впливати на основний агент.

---

## 9. Що означають результати

Після `deepeval test run tests/` ви бачите таблицю:

```
┃ Test case           ┃ Metric               ┃ Score ┃ Status ┃
┃ test_critic_revises ┃ Critique Quality      ┃ 1.0   ┃ PASSED ┃
┃ test_researcher_gr… ┃ Groundedness [GEval]  ┃ 0.1   ┃ FAILED ┃
```

| Score | Інтерпретація |
|---|---|
| 0.9–1.0 | Відмінно — агент виконує задачу ідеально |
| 0.7–0.9 | Добре — є незначні вади |
| 0.5–0.7 | Прийнятно — є над чим працювати |
| < 0.5 | Провал — потрібно переглянути промпт або логіку |

**Підсумкова статистика:**
```
» Pass Rate: 82.35% | Passed: 14 | Failed: 3
» time taken: 1406.86s | token cost: $0.18 USD
```

> Час і вартість допомагають розуміти, наскільки дорогим є одне тестування і де є простір для оптимізації.

---

## 10. Де відображається debug в тестах?

При запуску з `DEBUG=1` або `--agent-debug`, тести виводять ті самі рядки, що й `debug on` в REPL:

```
[test_critic] → critique(request='Compare naive RAG...')
  findings snippet: '# RAG Approaches: Naive vs Advanced...'
[test_critic] ← critique returned (3484 chars)
  verdict: REVISE
  is_complete: False, is_fresh: True
  revision_requests: ['Refocus the structure...', ...]

[e2e] supervisor → plan()
  🔧 knowledge_search("naive RAG sentence-window retrieval...")
  📎 [4 documents found]
  🔧 web_search("sentence-window retrieval RAG benchmark")
  📎 [5 results found]
[e2e] supervisor → critique()
[e2e] round 0: interrupt (save_report) → auto-approving
```

Цей вивід дозволяє точно знати, що пішло не так у кожному тесті — без запуску системи вручну.

---

## 11. Типові помилки та як їх виправляти

| Помилка | Причина | Рішення |
|---|---|---|
| `AssertionError: Metrics: Groundedness... failed` | Поріг занадто високий для гібридного пошуку | Знизити `threshold` або розширити `evaluation_steps` |
| `AssertionError: Expected Planner to call 'knowledge_search'` | Planner вирішив обійтися без KB (його право) | Перевіряти не сам факт виклику, а якість плану |
| `assert 'save_report' in all_names` | Запит надто простий — Supervisor відповів без інструментів | Використовувати складний багаточастинний запит |
| `AttributeError: ... has no attribute 'name'` | deepeval v3.9+ змінив API | Використовувати `getattr(metric, 'name', type(metric).__name__)` |
| Тест зависає на довго (>10 хв) | Ланцюжок агентів залучає багато веб-пошуків | Нормально для e2e, встановіть терпіння 😊 |
