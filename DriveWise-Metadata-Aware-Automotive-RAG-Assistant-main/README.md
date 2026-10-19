# DriveWise Verify

DriveWise Verify is a metadata-aware automotive RAG assistant that helps users ask questions, verify claims, compare models, and get requirement-based recommendations using official Hyundai brochure evidence.

The project does not answer from general automotive knowledge. It first filters the brochure library by brand and model, retrieves relevant chunks, reranks them with automotive section awareness, and then responds only from the retrieved evidence.

## Core Idea

Vehicle brochures contain official manufacturer information, but they are long and hard to search manually. DriveWise Verify turns those brochures into a structured local knowledge base with metadata such as:

* Brand
* Model
* Brochure name
* Brochure version
* Section
* Page number
* Source path
* Chunk ID

The generated brochure data is stored in:

```text
data/hyundai_brochures.json
```

## Interaction Modes

### 1. Ask About a Vehicle

Select a Hyundai model and ask a natural-language question.

Example:

```text
What safety and comfort features are available?
```

DriveWise retrieves only that model's brochure chunks and generates a grounded answer with source attribution.

### 2. Verify an Automotive Claim

Select a model and enter a claim from a dealer, website, advertisement, or social media.

Example:

```text
Hyundai Creta provides six airbags.
```

The claim verifier returns one of four verdicts:

* Supported
* Contradicted
* Partially Supported
* Not Found

Important rule: absence of brochure evidence is not treated as contradiction. If the brochure does not provide enough evidence, the verdict is `Not Found`.

### 3. Compare Two Vehicle Models

Select two models and ask for a feature comparison.

Example:

```text
Compare their safety and comfort features.
```

DriveWise runs two independent retrieval pipelines so evidence for one model is not mixed with evidence for the other. The result is shown as a structured comparison table with separate sources.

### 4. Requirement-Based Recommendation

Enter requirements such as family use, highway travel, safety, comfort, city driving, space, technology, performance, EV range, or charging.

Example:

```text
I need a car mainly for family highway trips with strong safety and comfort.
```

DriveWise ranks available models by brochure evidence strength and section alignment. It does not claim a universal "best car"; it explains which models appear more suitable based on official brochure evidence.

## Retrieval Flow

```text
Official brochure PDFs
-> PDF text extraction
-> Section-aware text classification
-> Overlapping brochure chunks
-> Metadata attachment
-> Structured JSON knowledge base
-> User mode selection
-> Metadata-first filtering
-> Local TF-IDF retrieval
-> Section-aware reranking
-> Evidence extraction
-> Mode-specific answer, verdict, comparison, or recommendation
-> Source attribution and quality signals
```

## Project Structure

```text
.
|-- app.py
|-- brochures/
|   |-- alcazar.pdf
|   |-- creta.pdf
|   `-- ...
|-- data/
|   `-- hyundai_brochures.json
|-- scripts/
|   |-- ingest_hyundai_brochures.py
|   `-- run_streamlit.cmd
|-- src/
|   `-- drivewise/
|       |-- assistant.py
|       |-- chunking.py
|       |-- evaluation.py
|       |-- generation.py
|       |-- library.py
|       |-- models.py
|       |-- retrieval.py
|       `-- text_utils.py
`-- requirements.txt
```

## Setup

Install the required packages:

```bash
python -m pip install -r requirements.txt
```

Run the app:

```bash
streamlit run app.py
```

On Windows, you can also use:

```bat
scripts\run_streamlit.cmd
```

## Updating Brochure Data

Place Hyundai brochure PDFs in the `brochures/` folder and run:

```bash
python scripts/ingest_hyundai_brochures.py
```

You can also provide a custom brochure folder:

```bash
python scripts/ingest_hyundai_brochures.py --folder C:\path\to\brochures
```

The ingestion script extracts text, infers automotive sections, attaches metadata, and writes the updated JSON library.

## Quality Signals

DriveWise Verify includes lightweight evaluation signals:

* Faithfulness
* Answer coverage
* Context relevance
* Claim verification confidence
* Recommendation evidence strength

These scores are simple local proxies, not external LLM judgments. They help users inspect whether a response is strongly grounded in the retrieved brochure evidence.
