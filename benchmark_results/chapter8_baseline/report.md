# TokenSmith Benchmark Report

**Run label:** `chapter8_baseline`  
**Generated:** 2026-04-26 00:21:19  
**QAC file:** `[PosixPath('synthetic_qac_data/manually_verified/Verified--chapter_08_qac_gemini.jsonl')]`  
**Questions evaluated:** 7  

---

## 📊 Executive Summary

> This section gives a plain-English overview of how well TokenSmith performed
> across all 7 benchmark questions.

| Metric | Score | Plain English |
| --- | --- | --- |
| 🔍 Gold Chunk Coverage | 97.6% ██████████ | Retrieval is **excellent** — almost all key information was retrieved |
| ✅ Answer Correctness (with ref) | 0.71 / 1.0 | **moderate** — answers were partially correct on average |
| 📋 Rubric Satisfaction | 96.4% | 96.4% of rubric criteria were fully met |
| 🎯 Retrieval Precision | 21.4% | 21.4% of retrieved chunks were useful |
| 🔒 Answer Faithfulness | 1.00 / 1.0 | How often answers stuck to retrieved info |

---

## ⚙️ Configuration

The following TokenSmith settings were used for this benchmark run:

```
  chunk_mode: recursive_sections
  chunk_overlap: 300
  chunk_size_in_chars: 2000
  disable_chunks: False
  embed_model: models/Qwen3-Embedding-4B-Q5_K_M.gguf
  embedding_model_context_window: 4096
  enable_history: False
  enable_topic_extraction: False
  ensemble_method: rrf
  extracted_index_path: data/extracted_index.json
  gen_model: models/qwen2.5-3b-instruct-q8_0.gguf
  hyde_max_tokens: 300
  max_gen_tokens: 400
  max_history_turns: 3
  metrics: ['all']
  num_candidates: 50
  output_mode: terminal
  page_to_chunk_map_path: index/sections/textbook_index_page_to_chunk_map.json
  ranker_weights: {'faiss': 1, 'bm25': 0, 'index_keywords': 0}
  rerank_mode: cross_encoder
  rerank_top_k: 5
  rrf_k: 60
  system_prompt_mode: baseline
  top_k: 10
  use_double_prompt: False
  use_golden_chunks: False
  use_hyde: False
  use_indexed_chunks: False
```

---

## 🔍 Retrieval Performance

> **What this measures:** For each question, the benchmark checks whether
> the specific sentences needed to answer it were actually retrieved by TokenSmith.
> A gold chunk is a verbatim sentence from the textbook that is necessary to answer
> the question. If TokenSmith retrieves chunks that contain those sentences,
> it had access to the right information.

### Overall Retrieval Metrics

| Metric | Value | Meaning |
| --- | --- | --- |
| Gold Chunk Coverage Rate | 97.6% | Fraction of required sentences found in retrieved chunks |
| Perfect Coverage (100%) | 85.7% | Questions where ALL gold chunks were retrieved |
| Retrieval Precision | 21.4% | Fraction of retrieved chunks that contained at least one gold chunk |
| Mean Retrieved Chunks | 10.00 | Average number of chunks TokenSmith retrieved per question |

### Retrieval by Difficulty

| Difficulty | N | Coverage Rate | Perfect Coverage | Precision |
| --- | --- | --- | --- | --- |
| Easy | 4 | 100.0% | 100.0% | 15.0% |
| Medium | 3 | 94.4% | 66.7% | 30.0% |

> **Note on difficulty:** Easy questions require 1-3 specific sentences. Medium questions
> need 2-10 sentences, possibly spread across the chapter. Hard questions require
> multiple concepts and reasoning, and typically have more gold chunks spread further apart.

---

## ✅ Answer Quality

> **What this measures:** Three complementary views of answer quality.
> (1) Whether the answer satisfies the evaluation rubric criteria.
> (2) An overall correctness score from -1 (wrong) to 1 (fully correct).
> (3) Whether the answer was faithful to the retrieved chunks.

### Rubric Satisfaction

The rubric for each question specifies key points a correct answer must address.

| Mode | Met Rate | Partial Rate | Avg Score |
| --- | --- | --- | --- |
| Individual (one call per criterion) | 96.4% | 96.4% | 0.96 |
| All-at-once (one call for all criteria) | 96.4% | 96.4% | 0.96 |

### Rubric Satisfaction by Difficulty

| Difficulty | N | Met Rate (Individual) | Met Rate (All) |
| --- | --- | --- | --- |
| Easy | 4 | 100.0% | 100.0% |
| Medium | 3 | 91.7% | 91.7% |

### Answer Correctness (-1 / 0 / 1)

- **1** = fully correct, addresses all key rubric points
- **0** = partially correct, addresses some rubric points
- **-1** = incorrect or completely off-topic

| Score | Without Reference | With Reference |
| --- | --- | --- |
| ✅ 1 (Fully correct) | 5 (71.4%) | 5 (71.4%) |
| ⚠️ 0 (Partially correct) | 2 (28.6%) | 2 (28.6%) |
| ❌ -1 (Incorrect) | 0 (0.0%) | 0 (0.0%) |
| Mean score | 0.71 | 0.71 |

### Answer Faithfulness

> **What this measures:** Did the answer claim anything that wasn't supported by
> the retrieved chunks? An unfaithful answer introduces facts or claims that the
> system had no basis for — a sign of hallucination.

| Verdict | Count | Percentage |
| --- | --- | --- |
| ✅ Faithful | 4 | 57.1% |
| ⚠️ Partially faithful | 0 | 0.0% |
| ❌ Unfaithful | 0 | 0.0% |
| ❓ Uncertain | 3 | 42.9% |

### Answer Length

| Metric | Value |
| --- | --- |
| Mean word count | 222.0 |
| Std dev word count | 93.1 |
| Mean char count | 1435 |

---

## 🎯 Chunk Relevance

> **What this measures:** Of the chunks TokenSmith retrieved, how many were
> actually relevant to answering the question? This is evaluated in two ways:
> individually (one judge call per chunk) and in groups of 3.

| Mode | Relevance Rate |
| --- | --- |
| Individual (per chunk) | 65.7% |
| Group (per 3 chunks) | 68.6% |

---

## 📚 Per-Chapter Breakdown

| Chapter | N | Gold Coverage | Rubric Met | Correctness | Faithfulness |
| --- | --- | --- | --- | --- | --- |
| Chapter 10 | 7 | 97.6% | 96.4% | 0.71 | 1.00 |

---

## 💡 Illustrative Examples

> These examples are selected automatically to illustrate both strong and weak
> performance. They are meant to give an intuitive feel for what the numbers mean.

### 🏆 Best Answer

**Chapter 10 | Difficulty: EASY | Record: `c10_w715_739_q000`**

**Question:**

> What are the three key metrics, often called the "three Vs," used to characterize Big Data and distinguish it from traditional relational databases?

**TokenSmith's Answer:**

> The three key metrics often used to characterize Big Data and distinguish it from traditional relational databases are:  
>   
> 1. **Volume**: The amount of data to be stored and processed is much larger than traditional databases were designed to handle. Traditional databases are typically optimized for handling smaller volumes of data.  
>   
> 2. **Velocity**: The speed at which data is generated and processed is much higher than what traditional databases were designed to handle. Traditional databases are …

**Key Metrics:**
- Gold Chunk Coverage: 100.0%
- Correctness (with ref): ✅ Fully correct
- Rubric Met Rate: 100.0%
- Faithfulness: ✅ Faithful

**Judge Explanation:**

> The student correctly identified and explained the three key metrics: Volume, Velocity, and Variety, matching the reference answer.

### 🔻 Worst Answer

**Chapter 10 | Difficulty: MEDIUM | Record: `c10_w715_739_q004`**

**Question:**

> How do key-value stores, also known as NoSQL systems, typically differ from traditional relational database systems in terms of features and capabilities?

**TokenSmith's Answer:**

> Key-value stores, often referred to as NoSQL systems, differ significantly from traditional relational database systems in terms of features and capabilities. Here are the key differences:  
>   
> 1. **SQL Support**: Traditional relational database systems like SQL databases support SQL, a declarative query language. Key-value stores, on the other hand, typically do not support SQL. This lack of SQL support makes application development more complex, as developers need to work with the raw data rather …

**Key Metrics:**
- Gold Chunk Coverage: 100.0%
- Correctness (with ref): ⚠️ Partially correct
- Rubric Met Rate: 100.0%
- Faithfulness: ✅ Faithful

**Judge Explanation:**

> The student answer addresses some key points but misses or gets others wrong, such as the lack of SQL support and secondary indexing, and does not fully explain the trade-offs for scalability.

### 🔍 Retrieval Examples

> The following show a case where retrieval worked well and one where it didn't.

#### ✅ Best Retrieval (all gold chunks found)

**Question:** What are the three key metrics, often called the "three Vs," used to characterize Big Data and distinguish it from traditional relational databases?

**Gold chunks needed (5):**
- `Such data, are characterized by their size, speed at which they are generated, and the variety of formats, are generical`
- `Big Data has been contrasted with traditional relational databases on the following metrics:`
- `Volume: The amount of data to be stored and processed is much larger than traditional databases, including traditional p`
- `Velocity: The rate of arrival of data are much higher in today's networked world than in earlier days.`
- `While much of today's data can be efficiently represented in relational form, there are many data sources that have othe`

**Coverage:** 100.0%

**First retrieved chunk:**

> The growth of the World Wide Web in the 1990s and 2000s resulted in the need to store and query data with volumes that far exceeded the enterprise data that relational databases were designed to manage. Although much of the user-visible data on the web in the early days was static, web sites generat

#### ❌ Worst Retrieval (gold chunks missed)

**Question:** For a social-networking application that needs to show a user new posts from all their friends, explain two alternative approaches to implement this functionality using a key-value store that does not support joins.

**Gold chunks needed (6):**
- `There are applications that conceptually require joins but implement the joins either in application code or by a form o`
- `For example, in a social-networking application, when a user connects to the system, the user should be shown new posts `
- `If the data about posts and friends is maintained in relational format, this would require a join.`
- `Instead of a join done in the database, the application code could implement the join by first finding the set of friend`
- `Another alternative is as follows: whenever a user u 0  makes a post, for each friend u i of the user, a message is sent`
- `When that user u i checks for updates, all data required to provide a summary view of posts by friends are available in `

**Coverage:** 83.3%

**First retrieved chunk:**

> Key-value stores do not provide any query processing facility, such as SQL language support, or even lower-level primitives such as joins. Many applications that use keyvalue stores can manage without query language support. The primary mode of data access in such applications is to  Page 1031 store

### ✅ Faithful Answer Example

**Question:** What are the three key metrics, often called the "three Vs," used to characterize Big Data and distinguish it from traditional relational databases?

**TokenSmith's Answer:**

> The three key metrics often used to characterize Big Data and distinguish it from traditional relational databases are:  
>   
> 1. **Volume**: The amount of data to be stored and processed is much larger than traditional databases were designed to handle. Traditional databases are typically optimized for handling smaller volumes of data.  
>   
> 2. **Velocity**: The speed at which data is generated and processe…

**Judge Explanation:**

> All claims in the answer are directly supported by the information provided in the retrieved text chunks.

### ✅ Rubric Pass Example

**Question:** What are the three key metrics, often called the "three Vs," used to characterize Big Data and distinguish it from traditional relational databases?

**Answer:**

> The three key metrics often used to characterize Big Data and distinguish it from traditional relational databases are:  
>   
> 1. **Volume**: The amount of data to be stored and processed is much larger than traditional databases were designed to handle. Traditional databases are typically optimized for handling smaller volumes of data.  
>   
> 2. **Velocity**: The speed at which data is generated and processe…

**Rubric criteria:**
- Must identify Volume as a key metric and explain it refers to the large amount of data.
- Must identify Velocity as a key metric and explain it refers to the high rate of data arrival.
- Must identify Variety as a key metric and explain it refers to the different formats of data beyond relational.

**Rubric met rate:** 100.0%

### ❌ Rubric Fail Example

**Question:** Describe the architecture of the Hadoop Distributed File System (HDFS) and explain the process for both reading from and writing to a file.

**Answer:**

> The architecture of the Hadoop Distributed File System (HDFS) is designed to efficiently store large files across a cluster of machines while providing a unified file system view to clients. The core of HDFS is a server known as the NameNode, which is responsible for managing the metadata of the file system. Here's a detailed explanation of the architecture and the process for reading from and wri…

**Rubric criteria:**
- Must identify the key components: NameNode and DataNodes.
- Must explain that files are broken into replicated blocks stored on DataNodes.
- Must describe the read process: client contacts NameNode for block locations, then contacts DataNodes for data.
- Must describe the write process: client contacts NameNode for block allocation, then sends data to DataNodes.

**Rubric met rate:** 75.0%

---

*Report generated by TokenSmith Benchmark Evaluator on 2026-04-26 00:21:19*
