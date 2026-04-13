# Retrieval Augmented Generation (RAG) Short Overview

> **⚠️ Best-effort draft:** this report was saved after reaching the maximum number of revise cycles and may still contain unresolved gaps noted by the Critic.

# Retrieval-Augmented Generation (RAG) Short Overview

## Executive summary

RAG (Retrieval‑Augmented Generation) — це патерн, у якому LLM перед відповіддю дістає релевантний контекст із зовнішнього сховища (векторна БД, пошуковий індекс, корпоративна база знань) і генерує відповідь, спираючись на ці дані. Це дозволяє додати до моделі «довгострокову памʼять», зменшити галюцинації, працювати з актуальною інформацією та уникнути дорогого fine‑tuning для кожного домену.

Нижче — короткий, практично орієнтований конспект: що таке RAG, з чого він складається, які інструменти використовують, як зібрати просту RAG‑систему та на що звертати увагу при оцінюванні й експлуатації.

---

## 1. Що таке RAG і навіщо він потрібен

**Визначення**  
Retrieval‑Augmented Generation (RAG) — це підхід, у якому LLM перед генерацією відповіді спочатку **дістає (retrieve)** релевантні фрагменти з зовнішнього джерела (база документів, БД, векторне сховище) і вже потім **генерує відповідь, спираючись на ці фрагменти**.

**Навіщо RAG:**
- дає доступ до **актуальних** даних, які не вміщені в параметрах моделі;
- дозволяє контролювати **джерела правди** (compliance, внутрішні політики, приватні дані);
- зменшує потребу у дорогому **fine‑tuning** для кожного домену;
- підвищує прозорість — відповідь можна **заземлити (ground)** на конкретні документи.

> Local KB: `retrieval-augmented-generation.pdf`, p.0–1 — визначення RAG, мотивація, роль зовнішнього сховища; `large-language-model.pdf`, p.7 — RAG як розширення LLM зовнішньою памʼяттю.

---

## 2. Базова архітектура і ключові компоненти

Архітектура ділиться на **офлайн‑індексацію** та **онлайн‑обробку запитів**.

### 2.1. Офлайн‑індексація даних

1. **Збір сирих даних**  
   PDF, HTML, DOCX, wiki/Confluence, Jira/ServiceNow тикети, БД.

2. **Нормалізація**  
   Приведення до єдиного формату: `text + metadata` (заголовок, дата, автор, тип, теги, ACL).

3. **Чанкінг (chunking)**  
   Розбиття документів на невеликі фрагменти:
   - логічні (по секціях/заголовках), або
   - фіксовані вікна (≈300–800 токенів з overlap 50–200).

4. **Ембеддінги й індексація**  
   - Для кожного чанка будуються **векторні подання (embeddings)**.
   - Вектори + метадані зберігаються у **векторне сховище** (vector DB).

### 2.2. Онлайн‑обробка запитів

1. **Query understanding / rewriting**  
   - попередня обробка запиту, за потреби — його переформулювання (LLM робить кращу/більш повну версію запиту, декомпозицію на підпитання тощо).

2. **Retriever (модуль пошуку)**  
   - приймає текст запиту;
   - повертає top‑k релевантних чанків з векторної/пошукової БД;
   - варіанти:
     - **sparse** (BM25, keyword search),
     - **dense** (vector search по ембеддінгах),
     - **hybrid** (комбінація sparse + dense + фільтри по метаданих).

3. **(Опційно) Реранкер**  
   - переоцінює top‑k результатів (часто cross‑encoder або LLM‑based reranker) і відбирає найкращі n (3–10) для промпта LLM.

4. **Generator (LLM)**  
   - отримує:
     - оригінальний запит,
     - відібраний контекст (чанки),
     - системні інструкції (мова відповіді, стиль, правила groundedness);
   - генерує відповідь, часто з:
     - цитатами з контексту,
     - посиланнями на джерела.

5. **Інфраструктурні компоненти**  
   - кеш відповідей і результатів retrieval;
   - логування й трейсинг;
   - моніторинг якості (метрики RAG, user feedback);
   - механізми безпеки (ACL, фільтри по метаданих).

> Local KB: `retrieval-augmented-generation.pdf`, p.1 — опис типового pipeline: ембеддінги, векторне сховище, кроки "query → retrieve → generate".

---

## 3. Типові інструменти й стек технологій (з tradeoffs RAG vs fine‑tuning)

### 3.1. Vector stores / БД

- **Faiss** — бібліотека (Meta) для високошвидкісного ANN‑пошуку, часто як індексний шар.
- **Chroma** — простий open‑source vector store, добре підходить для PoC.
- **Pinecone** — керована vector DB (SaaS) з масштабуванням, hybrid search, метаданими.
- **Weaviate** — open‑source + managed, підтримка hybrid search, GraphQL/REST.
- **pgvector** — розширення PostgreSQL для векторів та пошуку по відстані.
- Інші: **Qdrant, Milvus, Redisearch (vector), Elasticsearch/OpenSearch** з vector‑полями.

### 3.2. Фреймворки RAG / orchestration

- **LangChain** — ланцюжки (chains), агенти, інтеграції з LLM та vector DB, інструменти для RAG‑evaluation.
- **LlamaIndex** — індексація різних джерел, складні RAG‑графи, advanced індекси.
- **Haystack** — орієнтований на QA/RAG‑pipelines (retrievers, readers, evaluation).
- **Semantic Kernel** — оркестрація функцій/skills з RAG‑патернами (особливо для .NET/TypeScript).

### 3.3. Моделі

- **LLM**: GPT‑4.1/GPT‑4o, Claude, Llama 3, Mistral, Gemini тощо.
- **Embeddings**: OpenAI text‑embedding‑3‑small/large, Voyage, Cohere, SentenceTransformers, bge‑моделі.

### 3.4. Cost/performance: RAG vs fine‑tuning

**Коли логічніше RAG:**
- знання **часто оновлюються** (документація, політики, продукти);
- багато різних доменів, один LLM‑шар над усіма джерелами;
- важлива **прозорість джерел** (аудит, compliance);
- потрібен швидкий time‑to‑value з обмеженим бюджетом.

**Коли доречний fine‑tuning:**
- **вузький, стабільний домен**;
- задачі з жорстким **форматом виходу** (JSON, коди, шаблони договорів);
- **latency‑критичні** сценарії, де додатковий крок retrieval небажаний.

**Вплив вибору інструментів:**
- тип/налаштування векторного індексу (HNSW/IVF/Flat) + SaaS vs self‑hosted → latency/ціна;
- розмір і якість ембеддінгів та LLM → баланс між точністю, затримкою та вартістю.

> Web: огляди vector DB (Chroma, Pinecone, Weaviate, Qdrant, Milvus) і фреймворків (LangChain, LlamaIndex, Haystack) вказують на їх широку підтримку RAG‑сценаріїв.

---

## 4. Покроково: як побудувати просту RAG‑систему

Ціль: відповіді на питання по документації/PDF.

1. **Збір і нормалізація**  
   - зібрати всі релевантні документи (PDF/HTML/Markdown/wiki);
   - витягнути текст, уніфікувати формат, додати метадані (title, source, date, tags, ACL).

2. **Чанкінг**  
   - розбити документи на чанки 300–800 токенів з overlap 50–200;
   - не розривати таблиці, код, ключові списки посередині;
   - перевірити на прикладах, що типові питання можна покрити 1–2 чанками.

3. **Ембеддінги та індексація**  
   - вибрати модель ембеддінгів (бажано мультимовну/доменно адаптовану);
   - порахувати вектори для кожного чанка;
   - записати вектор + текст + метадані у vector DB (Chroma/Faiss/Pinecone/Weaviate/pgvector).

4. **Retriever**  
   - стартовий варіант: dense retrieval (cosine similarity/dot product, k ≈ 5–20);
   - покращення: hybrid пошук (BM25 + vector) + фільтрація по метаданих (language, domain, date).

5. **(Опційно) Реранкер**  
   - застосувати cross‑encoder/LLM‑reranker, щоб персортувати top‑k і віддати в LLM лише найкращі 3–5 чанків.

6. **Промптинг і генерація**  
   - системний промпт повинен:
     - вимагати спиратися тільки на наданий контекст,
     - зобовʼязувати вказувати, якщо інформації бракує,
     - за можливості — додавати посилання на джерела;
   - передавати в LLM: запит + список чанків (`[source, section, text]`).

7. **Інфраструктура**  
   - обгорнути pipeline в API/сервіс (наприклад, HTTP‑endpoint);
   - логувати: запит, retrieved контекст (ID, score), промпт, відповідь;
   - додати кеш для повторюваних запитів/відповідей;
   - налаштувати базовий моніторинг (latency, error rate) і збір прикладів для подальшого тюнінгу.

> Local KB: `retrieval-augmented-generation.pdf`, p.1 — типовий RAG‑ланцюг: індексація → векторне сховище → запит → пошук → генерація.

---

## 5. Як оцінювати якість RAG

Оцінювання RAG логічно розбити на дві частини:
1. **Retrieval** — чи дістаємо ми потрібні документи?
2. **Generation** — чи відповіді коректні, grounded і корисні?

### 5.1. Оцінка retrieval

**Датасет:** пари вигляду **(запит q → множина релевантних документів D\_relevant)**.  
Отримують через ручну розмітку, напів‑автоматично (LLM + людина), або синтетично з подальшою валідацією.

**Ключові метрики:**
- **Recall@k** — частка релевантних документів, що потрапили в топ‑k; критично, щоб не «пропускати» потрібне.
- **Precision@k** — частка релевантних серед top‑k; зменшує шум у контексті.
- **MRR (Mean Reciprocal Rank)** — наскільки рано в списку зʼявляється перший релевантний документ.
- **nDCG** — якість ранжування з урахуванням різних рівнів релевантності.
- **Coverage** — чи достатньо інформації в retrieved контексті, щоб відповісти на запит (часто оцінюється за допомогою LLM‑as‑a‑judge або вручну).
- RAG‑специфічні: `context_precision`, `context_recall`, `context_utilization` (Ragas/TruLens).

### 5.2. Оцінка generation

**Критерії:**
- **Faithfulness / groundedness** — відповідь не суперечить контексту, не вигадує фактів поза ним;
- **Factuality** — фактична правильність щодо «реального світу»;
- **Relevance** — відповідь по суті питання, без off‑topic;
- **Coverage/completeness** — чи охоплені важливі аспекти запиту;
- **Helpfulness/usefulness** — ясність, структура, практична користь.

**Методи:**
- **Reference‑based** (gold answers): порівняння з еталонними відповідями (семантична схожість, покриття ключових фактів);
- **Human eval**: експерти ставлять оцінки за шкалами (0–5) по faithfulness/relevance/helpfulness;
- **LLM‑as‑a‑judge**: інший LLM оцінює groundedness, hallucination rate, релевантність;
- **Бібліотеки**: 
  - **Ragas** — `faithfulness`, `answer_relevance`, `context_precision/recall`;
  - **TruLens** — groundedness, context/answer relevance + observability;
  - інші (DeepEval, Arize Phoenix) для CI/CD‑оцінювання.

> Web: Patronus/Pinecone/Unstructured блоги деталізують метрики faithfulness, groundedness, hallucination rate, а також описують бібліотеки Ragas і TruLens як стандартні інструменти.

---

## 6. Типові помилки та best practices

### 6.1. Типові помилки (failure modes)

- **Поганий чанкінг** — занадто великі (шум, перевищення контексту) або занадто дрібні чанки (розрив логіки, падіння recall).
- **Слабкий retriever / ембеддінги** — невідповідна модель ембеддінгів (мова/домен), відсутній hybrid search там, де потрібен.
- **Неправильний top‑k** — замалий (втрачаємо релевантні документи) або завеликий (шум, latency, ціна).
- **Ігнорування контексту LLM** — промпт не змушує посилатися на джерела/визнавати «не знаю», результат — галюцинації.
- **Помилки з метаданими/ACL** — недостатній контроль доступу до документів; неправильні фільтри за мовою/версією/клієнтом.
- **Відсутність моніторингу** — немає логів, метрик groundedness/hallucination rate, user feedback.
- **Latency/вартість** — надто великий LLM, завеликий top‑k, відсутність кешу.

### 6.2. Best practices

- **Структурований chunking** — орієнтація на логічні розділи, overlap, акуратне поводження з таблицями/кодом.
- **Hybrid retrieval + reranking** — поєднувати BM25 і dense search, додавати reranker для критичних запитів.
- **Сильний промптинг** — чіткі інструкції щодо використання лише контексту, визнання прогалин, цитування джерел.
- **Observability & continuous eval** — логувати повний ланцюжок, використовувати Ragas/TruLens, підтримувати golden‑set тестування.
- **Security & governance** — ACL/role‑based доступ у vector store, за потреби — маскування PII при індексації.
- **Кешування та оптимізація** — кеш відповідей і результатів retrieval, адаптивний top‑k, роутинг запитів на менші/дешевші моделі.

> Web: Unite.ai “How to Build Reliable RAG: Seven Failure Points…” описує подібні failure points (chunking, retriever, top‑k, моніторинг, latency/cost) і пропонує відповідні практики.

---

## 7. Додаткові ресурси для поглибленого вивчення

**Документація та туторіали:**
- LangChain docs — розділи «Retrieval» та «RAG» (приклади базового й advanced RAG, інтеграція з vector stores, evaluation).
- LlamaIndex docs — туторіали по RAG над PDF/wiki/БД, advanced індекси, секції «Evaluation».
- Haystack docs (deepset) — QA/RAG‑pipelines, retrievers, readers, evaluation nodes.
- Semantic Kernel docs — приклади semantic memory та RAG‑сценаріїв поверх Azure AI Search та інших сховищ.

**Огляди та статті:**
- Pinecone / Patronus / Unstructured — серії статей про архітектуру RAG, метрики, failure modes, best practices.
- Огляди vector DB (Chroma, Qdrant, Pinecone, Weaviate, Milvus та ін.) — landscape і вибір інструменту для RAG.

**Демо‑репозиторії:**
- Офіційні GitHub‑репо LangChain, LlamaIndex, Haystack — приклади «chat with your docs», multi‑source RAG, продакшн‑орієнтовані демо.
- Open‑source проєкти «doc‑QA over PDFs» і «enterprise search RAG» як референси для архітектури.

---

## Sources

- Local knowledge base:  
  - `retrieval-augmented-generation.pdf`, p.0–2 — визначення, pipeline, роль ембеддінгів і векторних сховищ.  
  - `large-language-model.pdf`, p.7 — RAG як зовнішня памʼять для LLM.

- Web (2024–2025):  
  - SashiDo blog “RAG Pipeline vs Traditional LLMs” — опис RAG‑ланцюга та порівняння з «голим» LLM.  
  - glukhov.org “Vector Stores for RAG Comparison” — порівняння Chroma, Pinecone, Weaviate, Qdrant, Milvus.  
  - lakefs.io “Best 17 Vector Databases for 2025” — landscape vector DB.  
  - Unite.ai “How to Build Reliable RAG: Seven Failure Points and Evaluation Frameworks”.  
  - Patronus.ai, Pinecone, Unstructured, Ragas/TruLens docs — матеріали по метриках та оцінюванню RAG‑систем.